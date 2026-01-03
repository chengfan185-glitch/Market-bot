# ml_model.py
"""
ML 模型封装：训练 / 保存 / 加载 / 在线推理
- 使用 scikit-learn 做示例：RandomForestClassifier (可替换)
- 输入特征由 PoolFeatures + CapitalFeatures + MarketContext 组合而成（可扩展）
- 输出为 Action enum（HOLD/LONG/SHORT）
- 可从 Mongo (collection of StrategyCase documents) 或 JSONL 训练

NOTE: ML dependencies (sklearn, numpy, pandas, joblib) are imported lazily to allow
the project to run in mock mode without these packages installed.
"""
import os
from typing import Tuple, Any, Dict, List, Optional
from domain.models.market_state import Action, StrategyCase, StrategySnapshot, PoolFeatures, CapitalFeatures, MarketContext
from datetime import datetime

MODEL_FILENAME_DEFAULT = "ml_decision_model.joblib"


def _import_ml_deps():
    """Lazy import of ML dependencies to allow running without them installed."""
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    return joblib, np, pd, RandomForestClassifier, StandardScaler


class MLDecisionModel:
    def __init__(self, model=None, scaler=None):
        # Lazy init: only import ML deps and create defaults when actually needed
        # This allows the class to be imported without ML dependencies installed
        self._model = model
        self._scaler = scaler
        self.is_fitted = False

    @property
    def model(self):
        """Lazy initialization of the model."""
        if self._model is None:
            _, _, _, RandomForestClassifier, _ = _import_ml_deps()
            self._model = RandomForestClassifier(n_estimators=100, random_state=42)
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def scaler(self):
        """Lazy initialization of the scaler."""
        if self._scaler is None:
            _, _, _, _, StandardScaler = _import_ml_deps()
            self._scaler = StandardScaler()
        return self._scaler

    @scaler.setter
    def scaler(self, value):
        self._scaler = value

    # ----------------------------
    # 特征工程：把一个 StrategyCase / Snapshot 转为向量
    # 这里采用一个简单的可解释特征组合：选择 target_pool 的 PoolFeatures 并 concat capital + market
    # ----------------------------
    @staticmethod
    def extract_features_from_snapshot(snapshot: StrategySnapshot, target_pool_id: Optional[str]) -> Dict[str, float]:
        """
        返回 dict: 特征名称 -> 数值
        如果 target_pool_id 为 None，使用 pool_features 中分数最高的池子作为 target
        """
        # pick pool
        pf = None
        if target_pool_id and target_pool_id in snapshot.pool_features:
            pf = snapshot.pool_features[target_pool_id]
        else:
            # choose best by relative_apy_rank
            pvals = list(snapshot.pool_features.values())
            if not pvals:
                # fallback zeros
                return {}
            pf = max(pvals, key=lambda x: x.relative_apy_rank)

        features = {}
        # pool features
        features["fee_to_gas_ratio"] = pf.fee_to_gas_ratio
        features["relative_apy_rank"] = pf.relative_apy_rank
        features["apy_trend_3h"] = pf.apy_trend_3h
        features["apy_trend_12h"] = pf.apy_trend_12h
        features["swap_trend"] = pf.swap_trend
        features["fee_trend"] = pf.fee_trend
        features["tvl_outflow_rate"] = pf.tvl_outflow_rate

        # capital features
        cap = snapshot.capital
        features["total_capital_usd"] = cap.total_capital_usd
        features["utilized_capital_usd"] = cap.utilized_capital_usd
        features["free_capital_ratio"] = cap.free_capital_ratio
        features["pool_return_variance"] = cap.pool_return_variance
        features["fee_stability_score"] = cap.fee_stability_score
        features["max_drawdown_7d_pct"] = cap.max_drawdown_7d_pct
        features["switch_success_rate"] = cap.switch_success_rate

        # market features
        m = snapshot.market
        features["gas_price_gwei"] = m.gas_price_gwei
        features["network_congestion_index"] = m.network_congestion_index
        features["native_price_usd"] = m.native_price_usd or 0.0

        return features

    # ----------------------------
    # 训练：接受 DataFrame 或 X,y arrays
    # ----------------------------
    def fit(self, X, y):
        """Train the model. X should be a pandas DataFrame, y should be a Series."""
        # Lazy import ML dependencies when actually training
        _, _, pd, _, _ = _import_ml_deps()
        X_num = X.fillna(0.0).astype(float)
        self.scaler.fit(X_num)
        Xs = self.scaler.transform(X_num)
        self.model.fit(Xs, y)
        self.is_fitted = True
        return self

    def predict(self, snapshot: StrategySnapshot, target_pool_id: Optional[str] = None) -> Tuple[Action, float]:
        """
        online predict: 返回 (Action, confidence_score)
        confidence_score: 对应预测类别的概率
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call load() or fit() first.")

        # Lazy import ML dependencies when actually predicting
        _, np, pd, _, _ = _import_ml_deps()

        feat = self.extract_features_from_snapshot(snapshot, target_pool_id)
        if not feat:
            return Action.HOLD, 0.0
        df = pd.DataFrame([feat])
        Xs = self.scaler.transform(df.fillna(0.0).astype(float))
        probs = self.model.predict_proba(Xs)[0]
        classes = self.model.classes_
        idx = int(np.argmax(probs))
        chosen = classes[idx]
        # classes are stored as strings of Action enum
        action = Action(chosen)
        confidence = float(probs[idx])
        return action, confidence

    def predict_proba(self, snapshot: StrategySnapshot, target_pool_id: Optional[str] = None) -> Dict[Action, float]:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        # Lazy import ML dependencies when actually predicting
        _, _, pd, _, _ = _import_ml_deps()

        feat = self.extract_features_from_snapshot(snapshot, target_pool_id)
        if not feat:
            return {Action.HOLD: 1.0}
        df = pd.DataFrame([feat])
        Xs = self.scaler.transform(df.fillna(0.0).astype(float))
        probs = self.model.predict_proba(Xs)[0]
        classes = self.model.classes_
        return {Action(c): float(p) for c, p in zip(classes, probs)}

    # ----------------------------
    # IO
    # ----------------------------
    def save(self, path: str = MODEL_FILENAME_DEFAULT):
        # Lazy import joblib when saving
        joblib, _, _, _, _ = _import_ml_deps()
        joblib.dump({"model": self._model, "scaler": self._scaler}, path)

    @classmethod
    def load(cls, path: str = MODEL_FILENAME_DEFAULT):
        # Lazy import joblib when loading
        joblib, _, _, _, _ = _import_ml_deps()
        data = joblib.load(path)
        inst = cls(model=data["model"], scaler=data["scaler"])
        inst.is_fitted = True
        return inst

    # ----------------------------
    # 从保存的 StrategyCase 文档（Mongo 或 JSONL）构建训练集
    # 支持：list of dicts (raw doc), 每个 doc 为 case.to_mongo_dict(...)
    # ----------------------------
    @staticmethod
    def build_dataset_from_case_docs(docs: List[Dict[str, Any]]) -> Tuple[Any, Any]:
        # Lazy import pandas when building dataset
        _, _, pd, _, _ = _import_ml_deps()

        rows = []
        ys = []
        for d in docs:
            # d expected to have fields: snapshot -> poolFeatures (camelCase or snake) and capital etc.
            # Normalize by using StrategyCase.from_dict to get proper types
            try:
                sc = StrategyCase.from_dict(d)
            except Exception:
                # best-effort attempt if doc already dict with camelCase
                sc = StrategyCase.from_dict(d)
            # target action: try to read sc.decision and map to Action
            lab = sc.decision
            try:
                act = Action(lab)
            except Exception:
                # fallback: map strings
                lab_up = str(lab).upper()
                if "LONG" in lab_up or "ENTER" in lab_up:
                    act = Action.LONG
                elif "SHORT" in lab_up or "EXIT" in lab_up:
                    act = Action.SHORT
                else:
                    act = Action.HOLD
            feat = MLDecisionModel.extract_features_from_snapshot(sc.snapshot, sc.target_pool_id)
            if not feat:
                continue
            rows.append(feat)
            ys.append(act.value)
        if not rows:
            return pd.DataFrame(), pd.Series(dtype=object)
        X = pd.DataFrame(rows)
        y = pd.Series(ys)
        return X, y