# California Housing Regression Comparison

캘리포니아 주택 가격 데이터셋을 사용해 주택 가격 중앙값(`MedHouseVal`)을 예측하는 회귀 분석 프로젝트입니다.  
분석은 `scikit-learn`의 California Housing 데이터셋을 기반으로 하며, 탐색적 데이터 분석(EDA), 이상치 및 논리 오류 처리, 지리적 파생 변수 생성, 스케일링, OLS/Ridge/Lasso 회귀 모델 비교까지의 전 과정을 포함합니다.

본 README는 다음 두 산출물을 바탕으로 작성되었습니다.

- `california_housing_regression_comparison.ipynb`: 데이터 불러오기, 전처리, 모델 학습, 평가 지표 산출 코드가 포함된 Jupyter Notebook
- `compariosn_report.pdf`: 분석 목적, EDA 결과, 전처리 근거, 모델별 해석, 최종 결론이 정리된 보고서  
  참고: 사용자가 요청한 파일명은 `comparison_report.pdf`였으나, 실제 폴더 내 PDF 파일명은 `compariosn_report.pdf`입니다.

## 1. 분석 목적

본 분석의 목적은 캘리포니아 지역의 인구 통계 및 지리적 특성 데이터를 활용하여 블록 그룹 단위의 주택 가격 중앙값을 예측하는 것입니다. 특히 단순한 예측 성능 비교에 그치지 않고, 다음 질문에 답하는 데 초점을 두었습니다.

1. 어떤 변수가 캘리포니아 주택 가격 예측에 가장 큰 영향을 주는가?
2. 위도와 경도 같은 지리 변수는 선형 회귀 모델에서 어떤 방식으로 작동하는가?
3. Ridge와 Lasso 같은 규제 회귀 모델은 기본 OLS 모델 대비 성능과 해석 측면에서 어떤 차이를 보이는가?
4. L1 규제와 L2 규제가 다중공선성이 강한 변수 조합을 어떻게 다르게 처리하는가?

최종적으로 OLS, Ridge, Lasso 세 모델의 예측 성능과 회귀 계수를 비교하여, 캘리포니아 주택 가격 예측에서 지리적 요인과 소득 요인이 어떤 역할을 하는지 분석했습니다.

## 2. 데이터셋 구성

분석에는 `sklearn.datasets.fetch_california_housing()`으로 제공되는 California Housing 데이터셋을 사용했습니다.

- 원본 샘플 수: 20,640개
- 원본 독립변수 수: 8개
- 타겟 변수: `MedHouseVal`
- 타겟 단위: 블록 그룹 내 주택 가격 중앙값, 단위는 10만 달러
- 결측치: 모든 변수에서 결측치 없음
- 데이터 타입: 모든 변수가 `float64` 형태의 연속형 수치 변수

| 구분 | 변수명 | 설명 |
|---|---:|---|
| 독립변수 | `MedInc` | 블록 그룹 내 가구 소득의 중앙값 |
| 독립변수 | `HouseAge` | 블록 그룹 내 주택 연령의 중앙값 |
| 독립변수 | `AveRooms` | 가구당 평균 방 개수 |
| 독립변수 | `AveBedrms` | 가구당 평균 침실 수 |
| 독립변수 | `Population` | 블록 그룹 내 인구 |
| 독립변수 | `AveOccup` | 가구당 평균 주거 인원 |
| 독립변수 | `Latitude` | 위도 |
| 독립변수 | `Longitude` | 경도 |
| 타겟 | `MedHouseVal` | 블록 그룹 내 주택 가격 중앙값 |

원본 데이터의 주요 기술통계는 다음과 같습니다.

| 변수 | 평균 | 중앙값 | 최솟값 | 최댓값 | 주요 관찰 |
|---|---:|---:|---:|---:|---|
| `MedInc` | 3.8707 | 3.5348 | 0.4999 | 15.0001 | 오른쪽 꼬리가 있는 소득 분포 |
| `HouseAge` | 28.6395 | 29.0000 | 1.0000 | 52.0000 | 52에서 상한 처리된 값 존재 |
| `AveRooms` | 5.4290 | 5.2291 | 0.8462 | 141.9091 | 비현실적으로 큰 극단치 존재 |
| `AveBedrms` | 1.0967 | 1.0488 | 0.3333 | 34.0667 | 일부 값은 방 개수보다 클 가능성 |
| `Population` | 1425.4767 | 1166.0000 | 3.0000 | 35682.0000 | heavy-tailed 분포 |
| `AveOccup` | 3.0707 | 2.8181 | 0.6923 | 1243.3333 | 극단적으로 큰 비현실적 값 존재 |
| `MedHouseVal` | 2.0686 | 1.7970 | 0.1500 | 5.0000 | 5.00001 근방 상한 처리값 존재 |

## 3. 탐색적 데이터 분석 결과

### 3.1 결측치 및 데이터 타입

노트북에서 `X.info()`와 `y.info()`를 확인한 결과, 20,640개 행 전체에서 모든 변수의 non-null count가 동일했습니다. 따라서 별도의 결측치 대체나 제거는 필요하지 않았습니다.

### 3.2 단일 변수 분포와 이상치

PDF 부록 7.1의 히스토그램 및 박스플롯에서는 다음 문제가 확인되었습니다.

- `MedHouseVal`은 5.0을 초과하는 값이 `5.00001`로 일괄 기록되어 있었습니다. 이는 실제 가격의 자연스러운 상한이라기보다 데이터셋 생성 과정에서 고가 주택을 상한 처리한 결과로 해석했습니다.
- `HouseAge`는 52년 이상 주택이 모두 52로 기록되어 있었습니다. 이 역시 특정 기준 이상을 같은 값으로 묶은 상한 처리로 판단했습니다.
- `AveRooms`의 최댓값은 약 141.91로 일반적인 가구당 평균 방 개수로 보기 어렵습니다.
- `AveBedrms`의 최댓값은 약 34.07이며, 일부 행에서는 침실 수가 전체 방 수보다 큰 논리적 모순이 발생할 수 있습니다.
- `AveOccup`의 최댓값은 약 1243.33으로, 일반적인 주거 단위의 평균 거주 인원으로 보기 어렵습니다.
- `Population`은 최댓값이 35,682로 매우 큰 오른쪽 꼬리를 가지며, 지역 단위 인구 특성상 로그 변환 후보로 판단했습니다.

이러한 이상치와 상한 처리값은 선형 모델이 왜곡된 패턴을 학습하게 만들 수 있으므로, 이후 전처리 단계에서 명시적으로 통제했습니다.

### 3.3 상관관계와 다중공선성

원본 상관관계 분석에서는 `MedInc`가 `MedHouseVal`과 0.69의 상관계수를 보여, 주택 가격 예측에서 가장 강한 단일 변수로 나타났습니다.

또한 다음 다중공선성 문제가 확인되었습니다.

- `AveRooms`와 `AveBedrms`의 상관계수는 0.85로 매우 높았습니다.
- `Latitude`와 `Longitude`의 상관계수는 -0.92로 매우 강한 음의 상관을 보였습니다.
- 원본 VIF 결과에서 `Latitude`는 559.87, `Longitude`는 633.71로 매우 높게 나타났습니다.
- `AveRooms`와 `AveBedrms`도 각각 45.99, 43.59로 높은 VIF를 보였습니다.

이 결과는 선형 회귀 계수가 불안정하게 커지거나, 같은 정보를 가진 변수 간에 설명력이 분산될 가능성을 시사합니다.

### 3.4 지리적 공간 패턴

위경도 기반 지도 시각화에서는 캘리포니아 해안선 주변, 특히 샌프란시스코, 실리콘밸리, 로스앤젤레스, 샌디에고 인근에 고가 주택이 밀집하는 패턴이 확인되었습니다.

보고서에서는 이를 `해안선 프리미엄`으로 해석했습니다. 단순한 위도와 경도 변수만으로는 해안 도시 접근성이나 주요 거점 주변의 가격 상승 효과를 충분히 설명하기 어렵기 때문에, 이를 보완하기 위한 지리적 파생 변수가 필요하다고 판단했습니다.

## 4. 데이터 전처리

전처리의 목적은 EDA에서 발견된 상한 처리, 극단치, 논리적 모순, 지리적 비선형성을 통제하여 선형 회귀 모델이 보다 안정적으로 학습되도록 하는 것입니다.

### 4.1 상한 처리값 제거

다음 조건을 적용했습니다.

```python
raw_data_clean = raw_data_clean[
    (raw_data_clean['HouseAge'] < 52) &
    (raw_data_clean['MedHouseVal'] <= 5.0)
]
```

처리 근거는 다음과 같습니다.

- `HouseAge == 52`는 실제로 52년인 주택뿐 아니라 52년 이상 주택이 모두 모인 상한값일 가능성이 큽니다.
- `MedHouseVal > 5.0` 또는 `5.00001` 근방 값은 고가 주택이 하나의 상한값으로 압축된 관측치입니다.
- 이러한 값을 그대로 학습하면 모델이 상한값 자체를 자연스러운 가격 패턴으로 오해할 수 있습니다.

### 4.2 IQR 기반 극단치 제거

`AveRooms`와 `AveOccup`에 대해 IQR 기반 필터링을 수행했습니다.

```python
target_cols = ['AveRooms', 'AveOccup']

for col in target_cols:
    Q1 = raw_data_clean[col].quantile(0.25)
    Q3 = raw_data_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    raw_data_clean = raw_data_clean[
        (raw_data_clean[col] >= lower_bound) &
        (raw_data_clean[col] <= upper_bound)
    ]
```

일반적으로 1.5 IQR을 사용할 수도 있지만, 본 분석에서는 분포의 오른쪽 꼬리가 길고 실제 지역 특성에 따른 큰 값도 일부 존재할 수 있으므로 3 IQR을 사용했습니다. 즉, 정상 데이터 손실을 줄이면서 매우 극단적인 값만 제거하는 방향을 택했습니다.

### 4.3 논리적 모순 제거

침실 수가 전체 방 수보다 클 수 없으므로 다음 조건을 적용했습니다.

```python
raw_data_clean = raw_data_clean[
    raw_data_clean['AveBedrms'] <= raw_data_clean['AveRooms']
]
```

또한 `AveRooms`의 극단치를 먼저 통제했기 때문에, `AveBedrms`의 극단적 문제도 일정 부분 함께 완화되었습니다.

### 4.4 지리적 파생 변수 생성

#### `lon_lat_mul`

위도와 경도를 단순히 독립된 좌표로만 사용하지 않고, 두 변수의 상호작용을 반영하기 위해 `Longitude * Latitude`를 계산한 `lon_lat_mul` 변수를 추가했습니다.

```python
raw_data_clean['lon_lat_mul'] = (
    raw_data_clean['Longitude'] * raw_data_clean['Latitude']
)
```

이 변수는 선형 회귀 모델이 위도와 경도가 결합해 만들어내는 복합적인 위치 효과를 일부 학습할 수 있도록 하기 위한 장치입니다.

#### `dist_to_coast`

보고서에서는 고가 주택이 주요 해안 도시 주변에 밀집한다는 점에 주목했습니다. 이를 정량화하기 위해 샌프란시스코, 로스앤젤레스, 샌디에고의 좌표를 기준으로 각 관측치에서 가장 가까운 해안 도시까지의 유클리드 거리를 계산했습니다.

```python
coastal_cities = {
    'SF': (37.7749, -122.4194),
    'LA': (34.0522, -118.2437),
    'SD': (32.7157, -117.1611)
}
```

`dist_to_coast`는 값이 작을수록 주요 해안 도시와 가까운 지역임을 의미합니다. 전처리 후 상관관계표에서 `dist_to_coast`와 `MedHouseVal`의 상관계수는 -0.43으로 나타났습니다. 즉, 주요 해안 도시와 가까울수록 주택 가격이 높은 경향이 확인되었습니다.

### 4.5 로그 변환

`Population`은 지역 단위 인구 특성상 오른쪽 꼬리가 긴 heavy-tailed 분포를 보였기 때문에 `np.log1p()` 변환을 적용했습니다.

```python
raw_data_clean1['Population'] = np.log1p(raw_data_clean1['Population'])
```

반면 `dist_to_coast`와 `MedInc`는 로그 변환 시 예측 성능이 떨어지는 것으로 확인되어 변환 대상에서 제외했습니다.

### 4.6 데이터 분할과 스케일링

전처리 이후 데이터는 다음과 같이 구성되었습니다.

- 최종 샘플 수: 18,285개
- 최종 독립변수 수: 10개
- 최종 타겟 변수 수: 1개
- 학습 데이터: 12,799개
- 테스트 데이터: 5,486개
- 분할 비율: train 70%, test 30%
- `random_state`: 42

전처리 이후에도 일부 변수는 완전한 정규분포를 따르지 않고 이상치 영향이 남아 있으므로, 평균과 표준편차 기반의 `StandardScaler`가 아니라 중앙값과 사분위수를 사용하는 `RobustScaler`를 적용했습니다.

```python
robust_scaler = RobustScaler()
X_train_scaled = robust_scaler.fit_transform(X_train)
X_test_scaled = robust_scaler.transform(X_test)
```

## 5. 전처리 후 데이터 특성

전처리 후 주요 기술통계는 다음과 같습니다.

| 변수 | 평균 | 중앙값 | 최솟값 | 최댓값 | 변화 |
|---|---:|---:|---:|---:|---|
| `MedInc` | 3.6948 | 3.4703 | 0.4999 | 13.1477 | 고소득 극단 일부 완화 |
| `HouseAge` | 27.0875 | 27.0000 | 1.0000 | 51.0000 | 52 상한값 제거 |
| `AveRooms` | 5.2387 | 5.1962 | 0.8462 | 10.6395 | 비현실적 극단치 제거 |
| `AveBedrms` | 1.0677 | 1.0475 | 0.3333 | 3.4111 | 논리 모순 및 극단치 완화 |
| `Population` | 1472.8235 | 1210.0000 | 3.0000 | 28566.0000 | 이후 로그 변환 적용 |
| `AveOccup` | 2.9453 | 2.8565 | 0.7500 | 5.8859 | 1000 이상 극단치 제거 |
| `MedHouseVal` | 1.9002 | 1.7190 | 0.1500 | 5.0000 | 상한 처리값 완화 |
| `dist_to_coast` | 0.8095 | 0.4884 | 0.0043 | 4.6469 | 해안 도시 접근성 변수 추가 |

전처리 후 상관관계에서는 다음 특징이 나타났습니다.

- `MedInc`와 `MedHouseVal`: 0.66
- `dist_to_coast`와 `MedHouseVal`: -0.43
- `AveOccup`과 `MedHouseVal`: -0.25
- `Latitude`와 `Longitude`: -0.93
- `Latitude`와 `lon_lat_mul`: 거의 -1에 가까운 강한 상관

전처리와 파생 변수 생성 이후 VIF는 오히려 일부 변수에서 더 커졌습니다.

| 변수 | VIF |
|---|---:|
| `MedInc` | 21.4185 |
| `HouseAge` | 9.1688 |
| `AveRooms` | 65.1272 |
| `AveBedrms` | 99.3843 |
| `Population` | 3.2127 |
| `AveOccup` | 20.1466 |
| `Latitude` | 40620.1905 |
| `Longitude` | 3868.0457 |
| `lon_lat_mul` | 20940.8515 |
| `dist_to_coast` | 4.4447 |

이는 `Latitude`, `Longitude`, `lon_lat_mul`이 서로 강하게 얽힌 변수이기 때문입니다. 이 다중공선성은 이후 Lasso 모델이 `Latitude` 계수를 0으로 축소하는 중요한 배경이 됩니다.

## 6. 모델링 방법

비교한 모델은 다음 세 가지입니다.

1. OLS 다중선형회귀
2. Ridge Regression
3. Lasso Regression

### 6.1 OLS 다중선형회귀

OLS는 별도의 하이퍼파라미터 없이 오차제곱합을 최소화하는 기본 선형 회귀 모델입니다.

```python
ols_reg = LinearRegression()
ols_reg.fit(X_train_scaled, y_train)
```

### 6.2 Ridge Regression

Ridge는 L2 규제를 사용하는 선형 회귀 모델입니다. 분석에서는 `GridSearchCV`를 사용해 5-Fold 교차검증으로 최적의 `alpha`를 탐색했습니다.

- 탐색 범위: `0.0001`부터 `100`까지 `0.5` 간격
- 최적 alpha: 0.0001
- Test RMSE: 0.5773

최적 alpha가 0.0001로 거의 0에 가까웠기 때문에, Ridge 모델은 실질적으로 OLS와 거의 동일하게 작동했습니다.

### 6.3 Lasso Regression

Lasso는 L1 규제를 사용하는 선형 회귀 모델입니다. Ridge와 동일하게 `GridSearchCV`로 최적 `alpha`를 탐색했습니다.

- 탐색 범위: `0.0001`부터 `100`까지 `0.5` 간격
- 최적 alpha: 0.0001
- Test RMSE: 0.5788

Lasso 역시 최적 alpha는 0에 가까웠지만, L1 규제의 특성상 일부 변수의 계수가 정확히 0으로 축소되었습니다. 특히 `Latitude`의 계수가 0이 된 점이 핵심 해석 포인트입니다.

## 7. 모델 성능 비교

테스트 데이터 기준 성능은 다음과 같습니다.

| 평가 지표 | OLS | Ridge (`alpha=0.0001`) | Lasso (`alpha=0.0001`) |
|---|---:|---:|---:|
| MAE | 0.4275 | 0.4275 | 0.4283 |
| MSE | 0.3333 | 0.3333 | 0.3350 |
| RMSE | 0.5773 | 0.5773 | 0.5788 |
| MAPE | 0.2718 | 0.2718 | 0.2706 |
| R2 | 0.6416 | 0.6416 | 0.6397 |

성능 해석은 다음과 같습니다.

- OLS와 Ridge의 테스트 성능은 사실상 동일합니다.
- Ridge의 최적 alpha가 0.0001로 매우 작아, L2 규제가 계수를 거의 변화시키지 못했습니다.
- Lasso는 RMSE, MAE, MSE, R2 기준으로 OLS/Ridge보다 아주 미세하게 낮은 성능을 보였습니다.
- 다만 Lasso는 MAPE가 0.2706으로 OLS/Ridge의 0.2718보다 약간 낮았습니다.
- 절대 오차 기준으로는 OLS와 Ridge가 우세하지만, 변수 선택과 계수 안정성 측면에서는 Lasso가 의미 있는 장점을 보였습니다.

## 8. 모델별 회귀 계수 비교

### 8.1 OLS 계수

| 변수 | 계수 |
|---|---:|
| `MedInc` | 0.857125 |
| `HouseAge` | 0.092367 |
| `AveRooms` | -0.147485 |
| `AveBedrms` | 0.077626 |
| `Population` | 0.023461 |
| `AveOccup` | -0.259603 |
| `Latitude` | 6.183152 |
| `Longitude` | -3.214848 |
| `lon_lat_mul` | 9.534636 |
| `dist_to_coast` | -0.166267 |

OLS에서는 `lon_lat_mul`, `Latitude`, `Longitude`가 가장 큰 계수 크기를 보였습니다. 이는 지리적 위치가 가격 예측에서 매우 강하게 작용한다는 점을 보여줍니다.

### 8.2 Ridge 계수

| 변수 | 계수 |
|---|---:|
| `MedInc` | 0.857131 |
| `HouseAge` | 0.092372 |
| `AveRooms` | -0.147491 |
| `AveBedrms` | 0.077626 |
| `Population` | 0.023462 |
| `AveOccup` | -0.259600 |
| `Latitude` | 6.179138 |
| `Longitude` | -3.213733 |
| `lon_lat_mul` | 9.529439 |
| `dist_to_coast` | -0.166256 |

Ridge 계수는 OLS와 거의 동일합니다. 이는 L2 규제 강도가 너무 작아 계수 축소 효과가 거의 발생하지 않았기 때문입니다.

### 8.3 Lasso 계수

| 변수 | 계수 |
|---|---:|
| `MedInc` | 0.865924 |
| `HouseAge` | 0.099107 |
| `AveRooms` | -0.155025 |
| `AveBedrms` | 0.077794 |
| `Population` | 0.024659 |
| `AveOccup` | -0.254582 |
| `Latitude` | 0.000000 |
| `Longitude` | -1.495193 |
| `lon_lat_mul` | 1.528402 |
| `dist_to_coast` | -0.150092 |

Lasso에서는 `Latitude`가 완전히 제거되었습니다. 반면 `lon_lat_mul`, `Longitude`, `MedInc`는 주요 변수로 유지되었습니다.

이 결과는 Lasso가 중복된 정보를 가진 변수 중 일부를 제거하는 L1 규제의 특성을 잘 보여줍니다. 특히 `Latitude`와 `lon_lat_mul`은 거의 완전한 수준의 상관을 보였기 때문에, Lasso는 `Latitude`를 제거하고 `lon_lat_mul`을 유지하는 방식으로 모델을 단순화했습니다.

## 9. 주요 결과 해석

### 9.1 규제 효과는 제한적이었다

Ridge와 Lasso 모두 최적 alpha가 0.0001로 산출되었습니다. 이는 현재 전처리된 데이터와 선형 모델 구조에서는 강한 규제가 필요하지 않았다는 뜻입니다.

특히 Ridge는 OLS와 성능 및 계수가 거의 동일했습니다. 보고서에서는 이를 현재 OLS 모델이 심각한 과적합 상태가 아니며, 전처리 과정에서 노이즈와 극단치가 어느 정도 통제되었기 때문으로 해석했습니다.

### 9.2 지리적 변수가 가장 강한 예측 요인으로 나타났다

OLS와 Ridge에서는 `lon_lat_mul`, `Latitude`, `Longitude`가 가장 큰 계수 크기를 보였습니다. Lasso에서도 `lon_lat_mul`과 `Longitude`는 주요 변수로 남았습니다.

이는 캘리포니아 주택 가격이 단순한 구조적 특성보다 위치에 크게 좌우된다는 점을 보여줍니다. 지도 시각화에서도 고가 주택은 해안선을 따라 밀집했으며, 특히 샌프란시스코, 실리콘밸리, 로스앤젤레스, 샌디에고 주변에서 높은 가격이 관찰되었습니다.

`dist_to_coast`의 계수가 세 모델 모두에서 음수인 점도 같은 맥락입니다.

- OLS: -0.166267
- Ridge: -0.166256
- Lasso: -0.150092

즉, 주요 해안 도시와의 거리가 멀어질수록 예측 주택 가격은 낮아지는 방향으로 작용합니다.

### 9.3 Lasso는 중복된 지리 변수를 제거했다

가장 주목할 결과는 Lasso에서 `Latitude`의 계수가 0이 되었다는 점입니다.

전처리 후 상관관계 분석에서 `Latitude`는 `lon_lat_mul`과 거의 -1에 가까운 강한 상관을 보였습니다. 따라서 `lon_lat_mul`이 이미 위도 정보의 상당 부분을 포함하고 있었고, Lasso는 중복 정보로 판단한 `Latitude`를 제거했습니다.

이것은 Lasso의 대표적인 장점인 변수 선택 효과를 보여줍니다. 성능은 OLS/Ridge보다 아주 조금 낮아졌지만, 모델은 더 단순하고 해석 가능한 형태가 되었습니다.

### 9.4 L1 규제와 L2 규제는 다중공선성을 다르게 처리했다

OLS와 Ridge에서는 지리 변수들의 계수가 매우 크게 나타났습니다.

- OLS `lon_lat_mul`: 9.534636
- OLS `Latitude`: 6.183152
- OLS `Longitude`: -3.214848
- Ridge `lon_lat_mul`: 9.529439
- Ridge `Latitude`: 6.179138
- Ridge `Longitude`: -3.213733

이와 달리 Lasso에서는 계수 범위가 훨씬 좁아졌습니다.

- Lasso `lon_lat_mul`: 1.528402
- Lasso `Longitude`: -1.495193
- Lasso `Latitude`: 0.000000

보고서에서는 이를 다음과 같이 해석했습니다.

- L2 규제는 계수를 부드럽게 줄이는 방식이지만, 이번 분석에서는 alpha가 너무 작아 Ridge가 OLS와 거의 같아졌습니다.
- L1 규제는 중복되거나 불필요한 변수의 계수를 정확히 0으로 만들 수 있습니다.
- 따라서 Lasso는 지리 변수 간 다중공선성으로 인한 계수 팽창을 일부 억제했습니다.
- 그 결과 `Latitude`를 제거하고 `lon_lat_mul`, `Longitude`, `MedInc` 중심으로 가중치를 재분배했습니다.

### 9.5 Lasso의 trade-off

Lasso는 변수 선택과 계수 압축 측면에서 더 안정적인 구조를 보였지만, 절대 오차 기준 성능은 아주 미세하게 하락했습니다.

- RMSE: OLS/Ridge 0.5773, Lasso 0.5788
- R2: OLS/Ridge 0.6416, Lasso 0.6397
- MAPE: OLS/Ridge 0.2718, Lasso 0.2706

즉, Lasso는 일부 정보를 제거하면서 절대 예측 오차는 조금 증가했지만, 실제값 대비 비율 오차는 오히려 소폭 개선되었습니다. 보고서에서는 이를 변수 단순화로 인한 정보 손실 비용과 예측 안정성 사이의 trade-off로 해석했습니다.

## 10. 최종 결론

본 분석에서는 OLS, Ridge, Lasso 회귀 모델을 활용하여 캘리포니아 주택 가격을 예측하고 모델별 성능과 계수 차이를 비교했습니다.

주요 결론은 다음과 같습니다.

1. 원본 데이터에는 상한 처리값, 비현실적 극단치, 논리적 모순 데이터가 존재했으며, 이를 제거하거나 완화하는 전처리가 필요했습니다.
2. 지도 시각화와 파생 변수 분석 결과, 캘리포니아 주택 가격에는 해안 도시 접근성과 위치 효과가 강하게 작용했습니다.
3. `MedInc`는 원본 및 전처리 후 데이터 모두에서 주택 가격과 강한 양의 상관을 보이는 핵심 변수였습니다.
4. OLS와 Ridge는 성능과 계수 측면에서 거의 동일했으며, 이는 Ridge의 최적 규제 강도 `alpha=0.0001`이 사실상 규제가 거의 없는 수준이었기 때문입니다.
5. Lasso는 `Latitude`를 제거하고 지리 변수의 계수 팽창을 줄이는 방식으로 더 단순한 모델을 만들었습니다.
6. 절대 오차 기준으로는 OLS와 Ridge가 가장 우수했지만, 변수 선택과 다중공선성 통제까지 고려하면 Lasso가 안정적이고 균형 잡힌 모델로 해석될 수 있습니다.
7. 세 모델 모두 선형 모델이므로, 캘리포니아 부동산 가격의 복잡한 공간적 비선형성을 완전히 학습하기에는 한계가 있습니다.

보고서의 최종 판단은 다음과 같이 요약할 수 있습니다.

- 예측 성능만 보면 OLS와 Ridge가 가장 우수합니다.
- 해석 가능성과 변수 선택까지 고려하면 Lasso가 더 안정적인 선택입니다.
- 향후 성능 개선을 위해서는 Random Forest, Gradient Boosting, XGBoost, LightGBM 같은 비선형 모델이나 공간 정보를 더 정교하게 반영한 모델이 필요합니다.

## 11. 한계 및 개선 방향

### 11.1 선형 모델의 한계

캘리포니아 주택 가격은 지역, 해안 접근성, 도시권, 소득 수준, 인구 밀도 등이 복합적으로 작용하는 문제입니다. 특정 지역의 가격이 급격히 상승하는 공간적 비선형성이 존재하기 때문에, 단순 선형 회귀 계수만으로 모든 패턴을 설명하기 어렵습니다.

### 11.2 지리 변수의 다중공선성

`Latitude`, `Longitude`, `lon_lat_mul`은 서로 강하게 상관되어 있습니다. 이로 인해 OLS와 Ridge에서는 계수 스케일이 크게 팽창했습니다. Lasso가 이를 일부 완화했지만, 위치 정보를 더 좋은 방식으로 표현하는 추가적인 feature engineering이 필요합니다.

가능한 개선 방향은 다음과 같습니다.

- 해안선까지의 실제 거리 계산
- 주요 대도시별 거리 변수 추가
- 위경도를 클러스터링하여 지역 그룹 변수 생성
- 공간 격자 기반 feature 생성
- 위경도 기반 비선형 모델 적용

### 11.3 모델 확장

향후에는 다음 모델과 비교하면 분석의 완성도를 높일 수 있습니다.

- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost
- LightGBM
- CatBoost
- 공간 가중 회귀 또는 지리적 가중 회귀

특히 트리 기반 앙상블 모델은 변수 간 비선형 상호작용을 자동으로 포착할 수 있으므로, 본 분석에서 한계로 지적된 공간적 복잡성을 더 잘 학습할 가능성이 높습니다.

## 12. 실행 방법

### 12.1 필요 라이브러리

노트북에서 사용한 주요 라이브러리는 다음과 같습니다.

```python
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
import plotly.express as px
from statsmodels.stats.outliers_influence import variance_inflation_factor
```

### 12.2 분석 재현 순서

1. `california_housing_regression_comparison.ipynb`를 실행합니다.
2. `fetch_california_housing()`으로 데이터를 불러옵니다.
3. 원본 데이터의 히스토그램, 박스플롯, 상관관계, VIF를 확인합니다.
4. 상한 처리값과 극단치를 제거합니다.
5. `lon_lat_mul`, `dist_to_coast` 파생 변수를 생성합니다.
6. `Population`에 로그 변환을 적용합니다.
7. train/test 데이터를 7:3으로 분리합니다.
8. `RobustScaler`로 독립변수를 스케일링합니다.
9. OLS, Ridge, Lasso 모델을 학습합니다.
10. MAE, MSE, RMSE, MAPE, R2로 모델 성능을 비교합니다.
11. 모델별 회귀 계수와 feature coefficient plot을 해석합니다.

## 13. 산출물 요약

| 파일 | 내용 |
|---|---|
| `california_housing_regression_comparison.ipynb` | 전체 분석 코드, EDA, 전처리, 모델링, 평가 지표 산출 |
| `compariosn_report.pdf` | 과제 제출용 분석 보고서, 결과 해석, 부록 그래프 |
| `README.md` | 노트북과 PDF 내용을 통합한 프로젝트 설명 및 분석 보고서 |

