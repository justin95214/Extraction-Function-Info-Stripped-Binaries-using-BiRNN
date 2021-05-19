# N-byte 기법

### 1. 기존 불균형데이터 전처리 솔루션을 보완한, 시퀀스별 데이터 재구성기법

- 오버샘플링과 Under Sampling방식은 시퀀스별로 무작위성을 가져야 하는데, 바이너리별로 무작위성이 부여되면서 순서가 임의로 섞이게 되면 RNN의 의미가 사라진다고 판단함
- Weight balancing을 적용 결과, 학습데이터가 너무 극한 불균형데이터여서, 효과가 미비했음

◆ 기존 Imbalanced 솔루션

▲ Over Sampling, Under Sampling 방식
- Under Sampling : 높은 비중을 차지하는 클래스의 값들을 임의로 제거하는 방법
- Over Sampling : 소수 클래스의 값들을 복제하여, 그 수를 늘리는 방법

▲ Weight balancing 방식
- 모델을 훈련하는 동안 소수 클래스와 다수 클래스에, 클래스의 비율에 대해 가중치를 달리 두는 방법

## 2.컴파일러 Ver GCC complier 6~9 사용

- 16.04의 기본 GCC 컴파일러 default 버전이 V5.3.1이므로 Gcc 6부터 최근에 나온 Gcc 9버전까지 통계 실험을 하는 데 사용함

- GCC 컴파일러별, 최적화 옵션별 데이터 크기, 함수의 특징

![N-byte%20%E1%84%80%E1%85%B5%E1%84%87%E1%85%A5%E1%86%B8%2007432305420b4a5ba50090c857d3eea1/Untitled.png](https://github.com/justin95214/Extraction-Function-Info-Stripped-Binaries-using-BiRNN/blob/master/Extraction_img/Untitled.png)

## 추출한 데이터 특징

◆ **라벨링**
위 1. 부분에 따라, 다양한 HEX 바이너리를 추출하여 사용함 함수 타입인 바이너리 부분만 추출하여, 함수 시작 부분과 아닌 나머지를 1과 0으로 라벨링 진행함

◆ **불균형데이터**
시작과 나머지 부분이 라벨링이 다르므로 99% 이상이 나머지인 0으로 라벨링 1% 미만이 1인 함수 시작 정보로 라벨링 하여, 불균형데이터 형상이 띄게 됨

◆ **평가지표 F1-Score 기준**
실제로 함수 시작 정보 1 라벨링에 대한 실제 예측 비율은 낮게 나옴 그래서 Recall 값과 Precision
값을 조화평균으로 나타낸 F1-Score으로, 평가의 기준으로 세움

◆ **최적화 옵션별로 학습데이터 구성**

- 최적화 옵션별 함수 시작 정보 패턴 수

![N-byte%20%E1%84%80%E1%85%B5%E1%84%87%E1%85%A5%E1%86%B8%2007432305420b4a5ba50090c857d3eea1/Untitled%201.png](https://github.com/justin95214/Extraction-Function-Info-Stripped-Binaries-using-BiRNN/blob/master/Extraction_img/Untitled%201.png)

Solarized dark             |  Solarized Ocean
:-------------------------:|:-------------------------:
![](https://github.com/justin95214/Extraction-Function-Info-Stripped-Binaries-using-BiRNN/blob/master/Extraction_img/Untitled%202.png)  |  ![](https://github.com/justin95214/Extraction-Function-Info-Stripped-Binaries-using-BiRNN/blob/master/Extraction_img/Untitled%203.png)


- 함수 타입이 아닌 바이너리를 넣기 전후 데이터 구성 모습

![N-byte%20%E1%84%80%E1%85%B5%E1%84%87%E1%85%A5%E1%86%B8%2007432305420b4a5ba50090c857d3eea1/Untitled%204.png](https://github.com/justin95214/Extraction-Function-Info-Stripped-Binaries-using-BiRNN/blob/master/Extraction_img/Untitled%204.png)
