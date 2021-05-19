# 데이터 추출

## 1. 스트립 바이너리

# **스트립된(stripped) 바이너리 분석의 어려움**
- 스트립된 바이너리에는 디버그 심볼이 포함되어 있지 않기 때문에 분석이 쉽지 않음
- 일반 바이너리와 스트립된 바이너리의 차이

### 데이터 - 바이너리

![Untitled (2).png](https://github.com/justin95214/Extraction-Function-Info-Stripped-Binaries-using-BiRNN/blob/master/Extraction_img/Untitled%20(2).png)

가. ELF 파일 형식인 리눅스, 유닉스 시스템의 표준 바이너리 파일 형식
나. 물체 파일과 링커를 거쳐서 나온 실행파일을 가지고 바이너리를 추출
다. ELF 파일은 ELF 헤더와 파일 데이터로 구성되어 있음
라. ELF 파일로부터 생성되는 플래시메모리에 다운로드 되는 16진수
로 인코딩된 파일이므로, 모든 헤더 테이블과 모든 부분을 16진수 로 추출할 수 있음
마. 실제로 짠 코드 부분은, 함수와 변수 등 여러 가지로, 함수 타입은
HEX 파일로 Text섹션에 주로 분포되어있음

## 2.바이너리 GCC컴파일러별 옵션별 유틸리지패키지 추출

![Untitled (4).png](https://github.com/justin95214/Extraction-Function-Info-Stripped-Binaries-using-BiRNN/blob/master/Extraction_img/Untitled%20(4).png)

## 3.컴파일러 Ver GCC complier 6~9 사용

- 16.04의 기본 GCC 컴파일러 default 버전이 V5.3.1이므로 Gcc 6부터 최근에 나온 Gcc 9버전까지 통계 실험을 하는 데 사용함

- GCC 컴파일러별, 최적화 옵션별 데이터 크기, 함수의 특징

![%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%20%E1%84%8E%E1%85%AE%E1%84%8E%E1%85%AE%E1%86%AF%20f21b61c8ab414ca2962f049535d464b7/Untitled%202.png](https://github.com/justin95214/Extraction-Function-Info-Stripped-Binaries-using-BiRNN/blob/master/Extraction_img/Untitled%20(5).png)

### 4.데이터 추출

원리 : 실행파일의 시작 주소와 프로그램 안에 함수들의 시작 주소와 함수 크기를 확인하여 바이너리를 추출

1. 리눅스 명령어 redelf, -l옵션을 통해 헤더 정보를 확인한다 LOAD 헤더 부분에 Offset과 가상주소와 물리적 주소를 확인하여, 실행파일의 시작 주소를 확인함

![%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%20%E1%84%8E%E1%85%AE%E1%84%8E%E1%85%AE%E1%86%AF%20f21b61c8ab414ca2962f049535d464b7/Untitled%203.png](https://github.com/justin95214/Extraction-Function-Info-Stripped-Binaries-using-BiRNN/blob/master/Extraction_img/Untitled%20(6).png)

2. 리눅스 명령어 objdump, -t옵션을 통해 Symbol Table을 확인한다 grep를 통해 함수 타입의 정
보만 가져온다 심볼테이블로 함수의 시작 주소와 섹션위치, 함수의 크기를 확인함

![%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%20%E1%84%8E%E1%85%AE%E1%84%8E%E1%85%AE%E1%86%AF%20f21b61c8ab414ca2962f049535d464b7/Untitled%204.png](https://github.com/justin95214/Extraction-Function-Info-Stripped-Binaries-using-BiRNN/blob/master/Extraction_img/Untitled%20(7).png)

3. 리눅스 명령어 hexdump를 통해, 함수의 시작 주소와 크기를 통해 바이너리를 16진수로 출력함

![%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%20%E1%84%8E%E1%85%AE%E1%84%8E%E1%85%AE%E1%86%AF%20f21b61c8ab414ca2962f049535d464b7/Untitled%205.png](https://github.com/justin95214/Extraction-Function-Info-Stripped-Binaries-using-BiRNN/blob/master/Extraction_img/Untitled%20(8).png)
