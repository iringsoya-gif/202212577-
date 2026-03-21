"""
Data Preprocessing — Pure NumPy Implementation
================================================
Min-Max Scaling & Z-Score Normalization (sklearn 미사용)
"""

import numpy as np


class MinMaxScaler:
    """
    Min-Max Scaling: x_scaled = (x - x_min) / (x_max - x_min)

    결과 범위: [0, 1]
    특징: 이상치에 민감, 범위가 명확할 때 유용
    """

    def __init__(self, feature_range: tuple = (0.0, 1.0)):
        self.feature_range = feature_range
        self.min_: np.ndarray = None
        self.max_: np.ndarray = None
        self.scale_: np.ndarray = None
        self.data_range_: np.ndarray = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        """학습 데이터로 min/max 계산"""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        self.data_range_ = self.max_ - self.min_
        # 범위가 0인 피처 처리 (상수 컬럼)
        self.scale_ = np.where(self.data_range_ == 0, 1.0, self.data_range_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """정규화 적용"""
        X = np.asarray(X, dtype=float)
        scalar_input = X.ndim == 0
        if X.ndim <= 1:
            X = X.reshape(-1, 1) if X.ndim == 1 else X.reshape(1, 1)

        lo, hi = self.feature_range
        X_scaled = (X - self.min_) / self.scale_
        X_scaled = X_scaled * (hi - lo) + lo
        # Clip to [0, 1] to handle out-of-range predictions
        X_scaled = np.clip(X_scaled, lo, hi)
        return X_scaled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """역변환: scaled → original"""
        X_scaled = np.asarray(X_scaled, dtype=float)
        lo, hi = self.feature_range
        X = (X_scaled - lo) / (hi - lo)
        return X * self.scale_ + self.min_


class StandardScaler:
    """
    Z-Score Normalization: x_zscore = (x - mean) / std

    결과: 평균 0, 표준편차 1 분포
    특징: 이상치에 강건, 정규분포 가정 모델에 적합
    """

    def __init__(self):
        self.mean_: np.ndarray = None
        self.std_: np.ndarray = None
        self.var_: np.ndarray = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.mean_ = X.mean(axis=0)
        self.var_ = X.var(axis=0)
        self.std_ = np.sqrt(self.var_)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def generate_employee_data(n_samples: int = 300, random_state: int = 42) -> np.ndarray:
    """
    현실적인 직원 데이터 생성
    - salary: 2,500만원 ~ 1.5억원 (연봉)
    - age: 22 ~ 65세

    Returns
    -------
    X : ndarray shape (n_samples, 2)  columns: [salary, age]
    """
    rng = np.random.RandomState(random_state)

    # 연봉: 정규분포 기반 (평균 6천만, std 2천만)
    salary_base = rng.randn(n_samples) * 20_000_000 + 60_000_000
    # 일부 고소득자 (상위 10%)
    high_earners = rng.choice(n_samples, size=n_samples // 10, replace=False)
    salary_base[high_earners] += rng.uniform(30_000_000, 90_000_000, size=len(high_earners))

    salary = np.clip(salary_base, 25_000_000, 150_000_000).reshape(-1, 1)

    # 나이: 정규분포 (평균 38세, std 10세)
    age = rng.randn(n_samples) * 10 + 38
    age = np.clip(age, 22, 65).reshape(-1, 1)

    return np.hstack([salary, age])


def compute_stats(X: np.ndarray, feature_names: list) -> dict:
    """데이터 통계 계산"""
    stats = {}
    for i, name in enumerate(feature_names):
        col = X[:, i]
        stats[name] = {
            "min": float(col.min()),
            "max": float(col.max()),
            "mean": float(col.mean()),
            "std": float(col.std()),
            "var": float(col.var()),
            "median": float(np.median(col)),
            "range": float(col.max() - col.min()),
        }
    return stats
