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
### 0. Reference paper review (~ 20.04.24) 
    - 1. (USENIX)Recognizing functions in binaries with neural networks_augsut 2015 
    - 2. Recognizing Functions in Binaries with Neural Networks 요약본
    - 3. BinComp: A stratified approach to compiler provenance Attribution
    - 4. Byteweight: Learning to Recongnize Functions in Binary Code
    - 5. Extracting Compiler Provenance from Program Binaries
    - 6. Neural Reverse Engineering of Stripped Binaryies

### 1. 데이터 추출(2020.07.06)
    - 리눅스의 유틸리티패키지 Binutils2.34, Coreutils-8.32 GCC컴파일러 버전별 6,7,8,9 추출 완료
    
### 2. gcc컴파일러버전별로 Symbol Table을 통한 함수시작과 나머지 바이너리 라벨링
    - 각 유틸리티패키지에 대해서, 각 gcc컴파일러버전별로 실행파일(ELF포멧형식) Symbol Table을 통한 함수시작 바이너리와 나머지 부분에 0과 1로 라벨링
    - 이 후에 바이너리과 라벨링 맵핑에 대한 Annotation파일을 구성
   
    
<hr>

### 3. Bidirectional RNN 구현 (1) 및 Stripped Binary 관련 공부 (~ 20.05.08)
    - 문서 : (20.05.08) Recognizing Functions in Binaries with Neural Networks 구현 (논문 추가 내용 정리)
    - 문서 : (20.05.08) Bidirectional RNN 구현 및 Stripped Binary 공부
    - 소스코드 : Preprocessing/Preprocessing_Save_hexByteCode.ipynb

<hr>

### 4. BIRNN 구현 (2)  (~ 20.06.05)
    - 문서 : (20.06.05) BIRNN 구현
    - BIRNN 구현 시도 -> Imbalanced Data 문제 발생
    - Imbalanced Data 문제 해결 방법으로 -> function context/chunck & cutting binaries
    - 소스코드 : RNN_by_All_Binary/BiRNN_gcc5_op0~2_by_All_Binary.ipynb

<hr> 

### 5. DATA Featuring
    - Insight (1) 컴파일러별 함수 시작점 주변 바이트들이 일정한 규칙성을 띔
        - ex) gcc6의 경우 함수 시작점 ~ 6개의 byte는 55, 6f, 72 .. 유사한 330쌍
        - ex) gcc7, 8, 9 도 비슷, 단 최적화 진행시 개수가 훨씬 더 늘어남
    - Insight (2) 데이터의 매우 극단적인 Imbalance data 문제
        - ex) 함수라 하면 함수시작점 1개의 byte 와 함수길이-1 만큼의 함수 시작점이 아닌 byte를 가지고 있음
    - Insignt (3) 최적화 옵션별 학습데이터 구성 다르게
        - ex) 함수 최적화 o0~o3별 학습데이터를 구성하지만, 최종 사용모델에서는 합쳐서 사용
    - Insight (4) 데이터 물리적 양
        - ex) 데이터를 바이트로 변환시 전체 파일이 2000만 ~ 3000만 바이트 가까이 됨. -> 물리적인 효율을 얻기 위해 N-Byte를 사용한 이유도 있음 (대략 전체 데이터의 1/3 효율)
        
    - 소스코드 : 함수주변바이트확인/
    
### 6. 불균형 데이터 문제 Imbalanced Data에 대한 솔루션 (~ 20.06.23)
    - 문서 : (20.06.23) Imbalanced Data에 대한 Solution 
    - 기존 Imbalance 해결법인 (오버샘플링, 언더샘플링 등은 RNN의 Sequential한 특징 때문에 적용이 힘듬)
    - Imbalance 해결보다는 완화하는 방식으로 생각 및 물리적 효율성을 위한 함수 시작점 주변 N-Byte 방식 생각**
    - 함수 시작 주변 N byte 자르는 방식 적용 및 다양한 N-Byte 기준 실험 진행
    - 함수의 시작 주변 양쪽 or 함수 시작점이 중간 등 여러가지 경우의 수 고려 (3byte, 10byte, 20byte ... 노가다 실험)
  

<hr> 

### 7. 최종 데이터 구성 전처리 (Preprocesing) (~ 20.07.07)
    - 문서 : (20.07.07)함수시작구분RNN
    - 전처리 1) 16진수 바이트를 One-Hot-Encoding을 작업
    - 전처리 2) 함수 시작 부분 부터 N-Byte 씩 잘라서 사용
    - 전처리 3) 함수 시작 부분 정보 포함 데이터 + 미포함데이터
      - 함수시작정보 포함데이터 + 미포함 데이터 5:5 비율 실험결과
    - 소스코드 : 함수시작정보포함데이터&정보미포함데이터
        
<hr>

### 8. 다양한 하이퍼 파라미터로 실험 진행 (~ 20.07.22)
    - 문서 : (20.07.22)바이너리 대상 컴파일러 및 함수정보 추출 기계학습 기술 연구(장두혁,김선민)
    - 실험결과 파일 : (20.07.22)국보연 실험(binutils).xlsx, (20.07.25)국보연 실험결과(coreutils).xlsx
    - N Byte 방식 다양한 하이퍼 파라미터로 실험 진행 -> 90% 결과 도달
    - N Byte 방식 GCC6 최적화 O0 ~ O3의 실험 진행 및 결과
    - 소스코드 : 함수시작정보포함데이터 & 정보미포함데이터_GCC6 이하 폴더들 
     
<hr> 

### 9. 다양한 하이퍼 파라미터로 실험 진행2 (~ 20.07.29) 
    - 문서 : (20.07.29)국보연 GCC6 결과최종
    - 실험결과 파일 : (20.07.27)국보연 실험결과(binutils&coreutils).xlsx
    - GCC 6 바이트추출시 CODE 부분만 데이터 재수집
    - 학습 데이터 재 추출 후 학습 진행 -> 99% 결과 도달
  - 소스코드 : gcc6_model, gcc7_model, gcc8_model, gcc9_model
      
<hr>

### 10. 바이너리 컴파일러별 옵션별 분류 탐지 모델과 함수 정보 추출 모델 pipline 실행프로그램 제작완료(~20.10.26) 
다운로드 링크
https://drive.google.com/drive/folders/1Ryfnt_CM2J8cL2yU2hB_xtr-P97viC9N?usp=sharing

### 11.  연구 종료 (~ 20.10.26)
  - kfold 에서 train (적당한 epoch 학습) + validation 으로 변경
  - 최적 hyperparameter 값 찾기 
  - 최적화 버전(o0 ~ o3) 각각 vs 최적화 통합 버전 비교 -> 최적화 통합 버전에서 조금더 좋은 결과를 보임
