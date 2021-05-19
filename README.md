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

### 0. Reference paper review (~ 20.04.24) 
    - 1. (USENIX)Recognizing functions in binaries with neural networks_augsut 2015 
    - 2. Recognizing Functions in Binaries with Neural Networks 요약본
    - 3. BinComp: A stratified approach to compiler provenance Attribution
    - 4. Byteweight: Learning to Recongnize Functions in Binary Code
    - 5. Extracting Compiler Provenance from Program Binaries
    - 6. Neural Reverse Engineering of Stripped Binaryies

### 1. 데이터 추출(1차: 2020.05.06 / 2차: 2020.06.23)
    - 리눅스의 유틸리티패키지 Binutils2.34, Coreutils-8.32 GCC컴파일러 버전별 6,7,8,9 추출 완료
    
### 2. gcc컴파일러버전별로 Symbol Table을 통한 함수시작과 나머지 바이너리 라벨링
    - 각 유틸리티패키지에 대해서, 각 gcc컴파일러버전별로 실행파일(ELF포멧형식) Symbol Table을 통한 함수시작 바이너리와 나머지 부분에 0과 1로 라벨링
    - 이 후에 바이너리과 라벨링 맵핑에 대한 Annotation파일을 구성
    - HEX 바이너리에 대해서 One-Hot Encoding을 통해서 벡터화
<hr>

### 3. Bidirectional RNN 구현 (1차:2020.05.08 / 2차: 20.06.05 )
    - 1차 : One-to-One RNN / BIRANN Model 구현
    - 2차 : Many-to-Many BIRNN Model구현

    
### 6. 불균형 데이터 문제 Imbalanced Data에 대한 솔루션 (~ 20.06.23)

  

<hr> 

### 7. 최종 데이터 구성 전처리 (Preprocesing) (~ 20.07.07)

        
<hr>

### 8. 다양한 하이퍼 파라미터로 실험 진행 (~ 20.07.22)

     
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
  
