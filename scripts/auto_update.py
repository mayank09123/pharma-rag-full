import sys, os, json, time, hashlib, requests, argparse, schedule
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv; load_dotenv()
from src.ingestion.fda_loader import FDALabelLoader
from src.retrieval.vector_store import DrugLabelVectorStore

TRACKER_FILE   = "./label_versions.json"
MONITORED_DRUGS = [
    "warfarin","metformin","lisinopril","aspirin",
    "atorvastatin","amoxicillin","ibuprofen","metoprolol",
    "amlodipine","omeprazole",
]

class LabelVersionTracker:
    def __init__(self):
        self.file     = TRACKER_FILE
        self.versions = json.load(open(self.file)) if Path(self.file).exists() else {}

    def save(self):
        with open(self.file,"w") as f: json.dump(self.versions, f, indent=2)

    def get(self, drug): return self.versions.get(drug, {})

    def update(self, drug, set_id, published_date, content_hash):
        self.versions[drug] = {
            "set_id": set_id, "published_date": published_date,
            "content_hash": content_hash,
            "last_checked": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat(),
        }
        self.save()

    def touch(self, drug):
        if drug in self.versions:
            self.versions[drug]["last_checked"] = datetime.utcnow().isoformat()
            self.save()

class AutoUpdater:
    def __init__(self):
        self.tracker = LabelVersionTracker()
        self.loader  = FDALabelLoader()
        self.store   = DrugLabelVectorStore()
        self.session = requests.Session()

    def get_latest(self, drug):
        try:
            resp = self.session.get(
                "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json",
                params={"drug_name": drug, "pagesize": 1}, timeout=15)
            data = resp.json().get("data", [])
            if not data: return {}
            return {"set_id": data[0].get("setid",""),
                    "published_date": data[0].get("published","")}
        except Exception as e:
            print(f"  ⚠ Cannot check {drug}: {e}"); return {}

    def get_hash(self, set_id):
        try:
            resp = self.session.get(
                f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{set_id}.xml",
                timeout=30)
            return hashlib.md5(resp.text.encode()).hexdigest()
        except: return ""

    def check_drug(self, drug):
        print(f"  Checking {drug}...")
        latest = self.get_latest(drug)
        if not latest: return {"drug": drug, "changed": False, "reason": "fetch failed"}
        stored = self.tracker.get(drug)
        if not stored:
            return {"drug": drug, "changed": True, "reason": "new drug",
                    "set_id": latest["set_id"]}
        if latest["set_id"] != stored.get("set_id",""):
            return {"drug": drug, "changed": True, "reason": "new label version",
                    "set_id": latest["set_id"]}
        if latest["published_date"] != stored.get("published_date",""):
            return {"drug": drug, "changed": True, "reason": "label updated",
                    "set_id": latest["set_id"]}
        cur_hash = self.get_hash(latest["set_id"])
        if cur_hash and cur_hash != stored.get("content_hash",""):
            return {"drug": drug, "changed": True, "reason": "content changed",
                    "set_id": latest["set_id"]}
        self.tracker.touch(drug)
        print(f"  ✓ {drug} — no changes")
        return {"drug": drug, "changed": False}

    def update_drug(self, drug, set_id):
        print(f"\n  Updating {drug}...")
        try:
            # Delete old chunks
            try:
                results = self.store._collection.get(
                    where={"drug_name": {"$contains": drug}},
                    include=["metadatas"])
                ids = results.get("ids", [])
                if ids:
                    self.store._collection.delete(ids=ids)
                    print(f"  Deleted {len(ids)} old chunks")
            except Exception as e:
                print(f"  ⚠ Could not delete old chunks: {e}")

            # Re-ingest
            chunks = self.loader.load_label_by_setid(set_id)
            if not chunks: return False
            self.store.add_chunks(chunks)

            # Update tracker
            latest = self.get_latest(drug)
            self.tracker.update(drug, set_id,
                latest.get("published_date",""),
                self.get_hash(set_id))
            print(f"  ✓ {drug} updated — {len(chunks)} chunks")
            return True
        except Exception as e:
            print(f"  ✗ Failed: {e}"); return False

    def run(self, drugs=None, check_only=False):
        drugs   = drugs or MONITORED_DRUGS
        start   = datetime.utcnow()
        print("\n" + "="*55)
        print("  FDA Drug Label Auto-Update")
        print(f"  {start.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("="*55)

        changed = []
        for drug in drugs:
            result = self.check_drug(drug)
            if result["changed"]: changed.append(result)
            time.sleep(0.5)

        if not changed:
            print("\n✅ All labels are up to date!"); return

        print(f"\n⚠ {len(changed)} update(s) found:")
        for c in changed: print(f"  • {c['drug']}: {c['reason']}")

        if check_only:
            print("\n(Check-only mode)"); return

        updated = []; failed = []
        for c in changed:
            ok = self.update_drug(c["drug"], c.get("set_id",""))
            (updated if ok else failed).append(c["drug"])

        print(f"\n{'='*55}")
        print(f"  ✓ Updated: {updated}")
        print(f"  ✗ Failed:  {failed}")
        print(f"  📊 Total chunks: {self.store.collection_stats()['total_chunks']}")

        report_path = f"update_report_{start.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump({"run_at": start.isoformat(),
                       "updated": updated, "failed": failed,
                       "changes": changed}, f, indent=2)
        print(f"  📄 Report: {report_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--schedule",   action="store_true")
    parser.add_argument("--interval",   default="daily",
                        choices=["hourly","daily","weekly"])
    parser.add_argument("--drug",       type=str, default=None)
    args = parser.parse_args()

    updater = AutoUpdater()

    if args.schedule:
        print(f"⏰ Scheduler started — {args.interval}")
        def job(): updater.run()
        if args.interval == "hourly":  schedule.every().hour.do(job)
        elif args.interval == "weekly": schedule.every().monday.at("08:00").do(job)
        else: schedule.every().day.at("08:00").do(job)
        job()
        while True: schedule.run_pending(); time.sleep(60)
    else:
        drugs = [args.drug] if args.drug else None
        updater.run(drugs=drugs, check_only=args.check_only)

if __name__ == "__main__":
    main()
