## ■ 기계학습 기반의 함수정보 구분 

### ■ Core Algorithm Skill
#### 1. 함수 정보 추출기술
#### 2. Class Imbalance Problem(Long-Tailed Problem)
#### 3. Bidirectional RNN model


### ■ 필요성 및 목표
    - 기계학습 기반의 스트립된(stripped) 리눅스 바이너리 분석 기술
    - 스트립 바이너리 제작에 사용된 컴파일러 탐지 기술
    - 스트립된 바이너리에서 함수 위치 탐지 기술

### ■ 연구 내용
    - 스트립된 리눅스 바이너리 제작에 사용된 컴파일러 정보 추출 기술 연구
    - 스트립된 리눅스 바이너리에서 함수 정보 추출 기술연구
    - 함수 위치정보(시작, 종료) 추출 machine learning 연구
    - 함수 Basic block 정보 추출 기술 연구

#### 키워드 #스트립된 바이너리 #바이너리 분석 #함수 정보 탐지
<hr>

### [0. Reference paper review (~ 20.04.24)](https://github.com/justin95214/Extraction-Function-Info-Stripped-Binaries-using-BiRNN/tree/master/Reference)
    - 1. (USENIX)Recognizing functions in binaries with neural networks_augsut 2015 
    - 2. Recognizing Functions in Binaries with Neural Networks 요약본
    - 3. BinComp: A stratified approach to compiler provenance Attribution
    - 4. Byteweight: Learning to Recongnize Functions in Binary Code
    - 5. Extracting Compiler Provenance from Program Binaries
    - 6. Neural Reverse Engineering of Stripped Binaryies
    
### [1. 데이터 추출(1차: 2020.05.06 / 2차: 2020.06.23 / 3차: 2020.07.28)](https://github.com/justin95214/Extraction-Function-Info-Stripped-Binaries-using-BiRNN/tree/master/Extraction%20Data)

## **스트립된(stripped) 바이너리 분석의 어려움**

가. 스트립된 바이너리에는 디버그 심볼이 포함되어 있지 않기 때문에 분석이 쉽지 않음
나. 일반 바이너리와 스트립된 바이너리의 차이

1. 스트립된 바이너리는 일반 바이너리와 달리 디버깅 정보가 들어간 디버그 심볼 정보가 제거된 형태로, 디스어셈블이나 역공학(reverse engineering)을 어렵게 함.
2. 일반 바이너리는 디버그 심볼 정보가 있기 때문에, 바로 모든 정보 조회가 가능함.
3. 스트립된 바이너리는 디버그 심볼 정보가 없어, 정보를 조회하는 데에 어려움을 겪음.
4. 스트립된 바이너리에서 사용된 컴파일러 종류와 버전 파악의 어려움
5. 스트립된 바이너리 내에 존재하는 함수 위치 파악의 어려움
    - 바이너리에는 함수에 포함되지 않는 부분도 존재함
    - 한 함수의 내용이 연속적이지 않을 수 있음
    - 호출되지 않는 함수가 존재할 수도 있음
    - 인라인되어 제거되는 함수도 있을 수 있음
    - 컴파일러 종류와 최적화 옵션에 따라 다르게 바이너리가 나옴, 특정 컴파일러나 최적화에 따라 함수 검출이 어려움

    ![Untitled.png](https://github.com/justin95214/Extraction-Function-Info-Stripped-Binaries-using-BiRNN/blob/master/Extraction_img/Untitled.png)


### [2. 리눅스의 유틸리티패키지 Binutils2.34, Coreutils-8.32 GCC컴파일러 버전별 6,7,8,9 추출 완료](https://github.com/justin95214/Extraction-Function-Info-Stripped-Binaries-using-BiRNN/tree/master/Extraction%20Data)
    - 1차 : header파일을 포함한 모든 바이너리를 추출
    - 2차 : 함수타입의 바이너리 98%이상을 .text섹션에서 추출
    - 3차 : 함수타입이외 모든 .text섹션의 바이너리를 추출
    
### 3. gcc컴파일러버전별로 Symbol Table을 통한 함수시작과 나머지 바이너리 라벨링
    - 각 유틸리티패키지에 대해서, 각 gcc컴파일러버전별로 실행파일(ELF포멧형식) Symbol Table을 통한 함수시작 바이너리와 나머지 부분에 0과 1로 라벨링
    - 이 후에 바이너리과 라벨링 맵핑에 대한 Annotation파일을 구성
    - HEX 바이너리에 대해서 One-Hot Encoding을 통해서 벡터화
    

### [4. Bidirectional RNN 구현 (1차:2020.05.08 / 2차: 20.06.05 )](https://github.com/justin95214/Extraction-Function-Info-Stripped-Binaries-using-BiRNN/tree/master/Model)
    - 1차 : One-to-One RNN / BIRANN Model 구현
    - 2차 : Many-to-Many BIRNN Model구현

### [5. 불균형 데이터 문제 Imbalanced Data(Long-Tailed Problem)에 대한 솔루션 N-byte기법 제시  (~ 20.06.23)](https://github.com/justin95214/Extraction-Function-Info-Stripped-Binaries-using-BiRNN/tree/master/Model)
  
### 6. 데이터 구성 전처리 (Preprocesing) (~ 20.07.07)
    - 1차 : input을 1개의단위로 One-Hot 인코딩만 진행
    - 2차 : input을 n개의 Window형식으로 전행
    - 3차 : mnay-2-many를 위한 n개씩 input을 위한 전처리 진행
    - 4차 : n-byte 기법 활용
        
### [7. 다양한 하이퍼 파라미터로 실험 진행 (~ 20.07.22)](https://github.com/justin95214/Extraction-Function-Info-Stripped-Binaries-using-BiRNN/tree/master/Document)
    - GCC컴파일러별 옵션별로 Input Sequence길이 최적점을 찾기위해서 통계적으로 실시
    - Hidden Layer의 너비와 깊이를 변경하면서 시도
     
      
<hr>

### 8. 바이너리 컴파일러별 옵션별 분류 탐지 모델과 함수 정보 추출 모델 pipline 실행프로그램 제작완료(~20.10.26) 
다운로드 링크
https://drive.google.com/drive/folders/1Ryfnt_CM2J8cL2yU2hB_xtr-P97viC9N?usp=sharing

### 10.  연구 종료 (~ 20.10.26)
  
