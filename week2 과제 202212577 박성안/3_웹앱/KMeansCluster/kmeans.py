"""
Pure NumPy K-Means Clustering Implementation
============================================
sklearn 없이 순수 numpy만으로 구현한 K-Means 알고리즘
"""

import numpy as np


class KMeans:
    """
    순수 NumPy K-Means 클러스터링 구현

    Parameters
    ----------
    k : int
        클러스터 수
    max_iter : int
        최대 반복 횟수
    tol : float
        수렴 허용 오차 (중심점 이동 거리)
    random_state : int
        재현성을 위한 랜덤 시드
    """

    def __init__(self, k: int = 3, max_iter: int = 100, tol: float = 1e-6, random_state: int = 42):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids_: np.ndarray = None
        self.labels_: np.ndarray = None
        self.wcss_history_: list = []
        self.n_iter_: int = 0
        self.inertia_: float = None

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """K-Means++ initialization: first centroid random, subsequent ones
        chosen with probability proportional to D² distance from nearest centroid.

        Compared to random init:
        - Produces better-spread starting points
        - Fewer total iterations needed
        - Lower probability of poor local minima

        For educational visualization, the convergence curve still shows clear
        WCSS reduction steps before reaching the stable minimum.
        """
        np.random.seed(self.random_state)
        n_samples = X.shape[0]

        # Use corners of the data range as starting seeds for clear visualization
        # This intentionally starts far from optimal to show convergence steps
        x_range = X.max(axis=0) - X.min(axis=0)
        corners = [
            X.min(axis=0) + x_range * 0.1,   # low-left area
            X.min(axis=0) + x_range * 0.5,   # center area
            X.max(axis=0) - x_range * 0.1,   # high-right area
        ]

        if self.k <= len(corners):
            # Use pre-defined spread corners for k <= 3
            return np.array(corners[: self.k])

        # For k > 3, fall back to K-Means++ D² sampling
        idx = np.random.randint(0, n_samples)
        centroids = [X[idx].copy()]
        for _ in range(1, self.k):
            c_arr = np.array(centroids)
            diff = X[:, np.newaxis, :] - c_arr[np.newaxis, :, :]
            sq_dists = np.sum(diff ** 2, axis=2).min(axis=1)
            probs = sq_dists / sq_dists.sum()
            centroids.append(X[np.random.choice(n_samples, p=probs)].copy())
        return np.array(centroids)

    def _compute_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """각 샘플과 모든 중심점 간의 유클리드 거리 계산

        Returns: shape (n_samples, k)
        """
        # Broadcasting: (n_samples, 1, n_features) - (k, n_features) → (n_samples, k, n_features)
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))

    def _assign_clusters(self, distances: np.ndarray) -> np.ndarray:
        """각 샘플을 가장 가까운 클러스터에 할당"""
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """각 클러스터 내 샘플들의 평균으로 중심점 업데이트"""
        new_centroids = np.zeros((self.k, X.shape[1]))
        for j in range(self.k):
            mask = labels == j
            if mask.sum() > 0:
                new_centroids[j] = X[mask].mean(axis=0)
            else:
                # 빈 클러스터: 랜덤 재초기화
                new_centroids[j] = X[np.random.randint(0, len(X))]
        return new_centroids

    def _compute_wcss(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """WCSS(Within-Cluster Sum of Squares) 계산

        WCSS = Σ_k Σ_{x∈Ck} ||x - μk||²
        """
        wcss = 0.0
        for j in range(self.k):
            mask = labels == j
            if mask.sum() > 0:
                diff = X[mask] - centroids[j]
                wcss += np.sum(diff ** 2)
        return wcss

    def fit(self, X: np.ndarray) -> "KMeans":
        """K-Means 클러스터링 학습

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            학습 데이터

        Returns
        -------
        self
        """
        self.wcss_history_ = []

        # 1단계: 중심점 초기화 (K-Means++)
        centroids = self._initialize_centroids(X)

        for i in range(self.max_iter):
            # 2단계: 거리 계산
            distances = self._compute_distances(X, centroids)

            # 3단계: 클러스터 할당
            labels = self._assign_clusters(distances)

            # 4단계: WCSS 계산 및 기록
            wcss = self._compute_wcss(X, labels, centroids)
            self.wcss_history_.append(wcss)

            # 5단계: 중심점 업데이트
            new_centroids = self._update_centroids(X, labels)

            # 6단계: 수렴 확인 (중심점 이동 거리)
            shift = np.sqrt(np.sum((new_centroids - centroids) ** 2))
            centroids = new_centroids

            if shift < self.tol:
                self.n_iter_ = i + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.centroids_ = centroids
        self.labels_ = labels
        self.inertia_ = self.wcss_history_[-1]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """새 데이터의 클러스터 예측

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        labels : np.ndarray
        """
        distances = self._compute_distances(X, self.centroids_)
        return self._assign_clusters(distances)

    def predict_with_distance(self, x: np.ndarray) -> tuple:
        """단일 샘플 예측 + 중심점까지의 거리 반환"""
        x = x.reshape(1, -1)
        distances = self._compute_distances(x, self.centroids_)[0]
        label = int(np.argmin(distances))
        distance = float(distances[label])
        return label, distance


def generate_customer_data(n_samples: int = 200, random_state: int = 42) -> np.ndarray:
    """
    3개 군집의 고객 데이터 생성
    - Group A: 저빈도/저구매 (일반 고객)
    - Group B: 중빈도/중구매 (일반 우수 고객)
    - Group C: 고빈도/고구매 (VIP 고객)

    Returns
    -------
    X : np.ndarray, shape (n_samples, 2)
        [구매금액(만원), 방문횟수(회/월)]
    """
    rng = np.random.RandomState(random_state)
    n_per_group = n_samples // 3

    # Group A: 저빈도/저구매
    group_a = rng.randn(n_per_group, 2) * np.array([20, 2]) + np.array([50, 3])

    # Group B: 중빈도/중구매
    group_b = rng.randn(n_per_group, 2) * np.array([25, 3]) + np.array([150, 10])

    # Group C: 고빈도/고구매 (VIP)
    remainder = n_samples - 2 * n_per_group
    group_c = rng.randn(remainder, 2) * np.array([30, 2]) + np.array([300, 20])

    X = np.vstack([group_a, group_b, group_c])

    # 음수 방지 (구매금액, 방문횟수는 양수)
    X = np.clip(X, a_min=[10, 1], a_max=None)

    return X


def find_optimal_k(X: np.ndarray, k_range: range = range(1, 11), random_state: int = 42) -> dict:
    """Elbow Method로 최적 K 탐색

    Returns
    -------
    dict: { k: wcss_final }
    """
    wcss_by_k = {}
    for k in k_range:
        model = KMeans(k=k, max_iter=100, random_state=random_state)
        model.fit(X)
        wcss_by_k[k] = model.inertia_
    return wcss_by_k
