# train_model.py
"""
训练脚本：
- 从 Mongo 集合或 JSONL 文件加载历史 StrategyCase 文档
- 构建特征矩阵并训练 MLDecisionModel
- 保存模型到磁盘 (joblib)
使用：
  python train_model.py --mongo-uri mongodb://... --db mydb --collection cases
或
  python train_model.py --jsonl data/cases.jsonl
"""
from market.adapters.market_feed import CEXFetcher
from domain.models.market_state import StrategySnapshot
from strategy.implementations.rule_strategy import RuleStrategy
from risk.implementations.basic_risk import BasicRisk
from execution.adapters.mock_broker import MockBroker


try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None

def load_docs_from_mongo(uri: str, db: str, collection: str, limit: int = None) -> List[Dict]:
    if MongoClient is None:
        raise RuntimeError("pymongo is required to load from Mongo")
    client = MongoClient(uri)
    coll = client[db][collection]
    docs = list(coll.find().limit(limit or 0))
    return docs

def load_docs_from_jsonl(path: str) -> List[Dict]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mongo-uri", type=str, default=None)
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--jsonl", type=str, default=None)
    parser.add_argument("--out", type=str, default="ml_decision_model.joblib")
    args = parser.parse_args()

    docs = []
    if args.mongo_uri and args.db and args.collection:
        docs = load_docs_from_mongo(args.mongo_uri, args.db, args.collection)
    elif args.jsonl:
        docs = load_docs_from_jsonl(args.jsonl)
    else:
        raise RuntimeError("Provide either --mongo-uri & --db & --collection or --jsonl")

    model = MLDecisionModel()
    X, y = MLDecisionModel.build_dataset_from_case_docs(docs)
    if X.empty:
        raise RuntimeError("No training data extracted.")
    print("Training samples:", len(X))
    model.fit(X, y)
    model.save(args.out)
    print("Saved model to", args.out)

if __name__ == "__main__":
    main()