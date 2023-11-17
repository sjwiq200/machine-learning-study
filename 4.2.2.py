import pandas as pd


# 데이터를 수집
df = pd.read_csv("https://raw.githubusercontent.com/wikibook/machine-learning/2.0/data/csv/basketball_stat.csv")

# 수집된 데이터 샘플을 확인.
print(df.head())

# 현재 데이터에서 포지션의 개수를 확인한다.
print(df.Pos.value_counts())


# 데이터 시각화
import matplotlib.pyplot as plt
import seaborn as sns

# 스틸, 2점슛 데이터 시각화.
# sns.lmplot(
#     x='STL', # x축
#     y='2P', # y축
#     data=df,  # 데이터
#     fit_reg=False,  # 노 라인
#     scatter_kws={"s": 150}, # 좌표 상의 점의 크기
#     markers=["o", "x"],
#     hue="Pos" # 예측값
# )

# # 타이틀
# plt.title('STL and 2P in 2d plane')
# plt.show()

# 분별력이 없는 특징(feature)을 데이터에서 제거
df.drop(['2P', 'AST', 'STL'],  axis=1, inplace = True)

print(df.head())

# 사이킷런의 ㅅrain_test_split 을 사용하면 코드 한 줄로 손쉽게 데이터를 나눌 수 있음.
from sklearn.model_selection import train_test_split

# 다듬어진 데이터에서 20%를 테스트 데이터로 분류.
train, test = train_test_split(df, test_size = 0.2)

print(train.shape[0])
print(test.shape[0])

# kNN 라이브러리 추가
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# 최적의 k를 찾기 위해 교차 검증을 수행할 k의 범위를 3부터 학습 데이터 절반까지 지정.
max_k_range = train.shape[0] // 2
k_list = []
for i in range(3, max_k_range, 2):
    k_list.append(i)
    
    
cross_validation_scores = []
# 학습에 사용될 속성을 지정
x_train = train[['3P', 'BLK', 'TRB']]
# 선수 포지션을 예측할 값으로 지정
y_train = train[['Pos']]

# 교차 검증 (10-fold)을 각 k를 대상으로 수행해 검증 결과를 저장
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train.values.ravel(), cv = 10, scoring='accuracy')
    cross_validation_scores.append(scores.mean())
    
print(cross_validation_scores)


# k에 따른 정확도를 시각화
plt.plot(k_list, cross_validation_scores)
plt.xlabel('the number of k')
plt.ylabel('Accuracy')
# plt.show()

# 가장 예측율이 높은 k를 선정
k = k_list[cross_validation_scores.index(max(cross_validation_scores))]
print("The best number of k : " + str(k))




'''
 최종
'''
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=k)

# kNN 모델 학습
knn.fit(x_train, y_train.values.ravel())

# 테스트 데이터에서 분류를 위해 사용될 속성을 지정
x_test = test[['3P', 'BLK', 'TRB']]
y_test = test[['Pos']]

# 테스트 시작
pred = knn.predict(x_test)

# 모델 예측 정확도(accuracy) 출력
print("accuracy : " + str(accuracy_score(y_test.values.ravel(), pred)) )

comparison = pd.DataFrame({'prediction': pred, 'ground_truth': y_test.values.ravel()})

print(comparison)
