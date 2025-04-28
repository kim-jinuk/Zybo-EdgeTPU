# Zybo-EdgeTPU **EO Real-Time Vision** Project 🚀  
30 FPS 임베디드 실시간 영상 개선·탐지·추적 파이프라인
[![Build](https://img.shields.io/github/actions/workflow/status/your-id/zybo_eo_rt/build.yml?branch=main)](../../actions)

---

## 1. 개요
Zybo Z7-10 보드와 **Google Coral USB Edge TPU**를 이용해  
**EO(주간) 카메라** 스트림을 실시간(≥30 FPS)으로  

1. ▶️ **프레임 캡처**  
2. ✨ **디블러링** & **초해상도**  
3. 🎯 **Edge TPU 객체 탐지** (MobileNet-SSD v2)  
4. 📍 **다중 객체 추적**(KCF/SORT) & 궤적 표시  
5. 💾 **주석 영상 표시·저장** (+ 이벤트 알림)  

까지 수행하는 **모듈형 파이프라인**입니다.

> - 실시간성 최우선 (딥러닝 모델 INT8 양자화)  
> - Python 컨트롤 + C++ 고속 OpenCV 연산 분리  
> - 모든 코드는 Linux / PetaLinux에서 동작 확인

---

## 2. 하드웨어 / 소프트웨어 스택
| 항목            | 사양 (권장)                       |
|-----------------|-----------------------------------|
| Board          | *Digilent Zybo Z7-10* (Zynq-7000) |
| Camera         | USB UVC or MIPI-CSI (720p @ 30 fps) |
| NPU            | **Google Coral USB Edge TPU**      |
| OS             | Ubuntu 20.04 ARM / PetaLinux 2023 |
| Python         | ≥ 3.9 (venv/conda 권장)           |
| C++            | ≥ 17,  GCC 11 +                    |
| OpenCV         | ≥ 4.8 (with `contrib`, BUILD_TFLITE ON) |
| TFLite Runtime | 2.15 (armhf/arm64)                |
| pycoral        | 2.0-post1                         |
| CMake / Ninja  | 3.18 + / 1.10 +                   |

---

## 3. 디렉터리 구조
``` text
.
├── README.md
├── .gitignore
├── requirements.txt                     # Python 의존성
├── setup_env.sh                         # 패키지 설치 스크립트 (pip + apt)
├── assets/
│   └── Image1.png                       # 데모 영상·이미지
├── docker/
│   └── Dockerfile                       # 로컬 개발/테스트용 컨테이너
├── cmake/                               # C++ 빌드 헬퍼
│   └── arm-linux-gnueabihf.cmake        # Zynq cross‑toolchain 설정
├── CMakeLists.txt                       # C++ 빌드를 최상위에서 관리
├── src/
│   ├── capture/
│   │   └── camera_capture.py            # Thread‑1: V4L2 / OpenCV 프레임 캡처
│   ├── python/
│   │   └── main.py                      # 데모 실행
│   ├── cpp/
│   │   └── CMakeLists.txt              # (예시)
│   ├── processing/
│   │   ├── deblurring.py               # DeblurGAN‑v2 Lite 추론 래퍼
│   │   └── super_resolution.py         # ESRGAN‑tiny 추론 래퍼
│   ├── detection/
│   │   └── tpu_detection.py            # Edge‑TPU MobileNet‑SSD 추론 래퍼
│   ├── tracking/
│   │   ├── sort_tracker.py             # IoU + 칼만필터(SORT) 구현
│   │   └── multi_tracker.py            # KCF/CSRT MultiTracker 래퍼
│   ├── pipeline/
│   │   ├── ouptut.py                   # Thread‑3: 디스플레이 & VideoWriter
│   │   └── pipeline.py                 # Thread‑2: 전체 파이프라인 조립
│   └── utils/
│       ├── config.py                   # YAML/JSON 설정 로더
│       └── logger.py                   # 공통 로깅 유틸
├── scripts/
│   ├── run_pipeline.py                 # 파이프라인 실행 엔트리
│   └── benchmark.py                    # FPS/Latency 벤치마크
└── build/                              # (CMake 아웃풋)
```

---

## 4. 빠른 시작

### 4-1. 저장소 클론
```bash
git clone https://github.com/kim-jinuk/Zybo-EdgeTPU.git
cd Zybo-EdgeTPU
git submodule update --init
```

### 4-2. Python 환경
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 4-3. C++ 모듈 빌드
```bash
mkdir build && cd build
cmake -GNinja ..
ninja && sudo ninja install      # libvision_core.so
```

### 4-4. 데모 실행
```bash
python src/python/main.py --source 0 \
       --model models/ssd_mobilenet_v2_edgetpu.tflite
```

기본값: 30 FPS, 640×480. \
Edge TPU 미검출 시 --cpu 옵션으로 강제 CPU 추론.

---

## 5. 핵심 모듈
| 모듈             | 파일/폴더                              | 주요 기능                                          |
| ---------------- | -------------------------------------- | ------------------------------------------------  |
| FrameCapture     | src/capture/camera_capture.py          | Thread‑1: V4L2 / OpenCV 프레임 캡처                |
| DeblurLite       | src/processing/deblurring.py           | DeblurGAN‑v2 Lite 추론 래퍼                        |
| SRLite           | src/processing/super_resolution.py     | ESRGAN‑tiny 추론 래퍼                              |
| EdgeTPUDetector  | src/detection/tpu_detection.py         | Edge‑TPU MobileNet‑SSD 추론 래퍼                   |
| Tracker          | src/tracking/sort_tracker.py           | IoU + 칼만필터(SORT) 구현                          |
| MultiTracker     | src/tracking/multi_tracker.py          | KCF/CSRT MultiTracker 래퍼                         |
| Pipeline         | src/pipeline/pipeline.py               | Thread‑2: 전체 파이프라인 조립                      |
| Output           | src/pipeline/output.py                 | Thread‑3: 디스플레이 & VideoWriter                 |
| Utils            | src/utils/                             | FPS 계측, 로그, 설정 파서 등                        |

각 모듈은 TODO: 주석으로 구현 포인트가 표시돼 있습니다.

---

## 6. 모델 준비
```bash
# COCO MobileNet-SSD Edge TPU 모델 다운로드 & 컴파일
wget https://dl.google.com/coral/canned_models/ssd_mobilenet_v2_coco_quant_postprocess.tflite -P models
edgetpu_compiler models/ssd_mobilenet_v2_coco_quant_postprocess.tflite
```

- 맞춤 클래스 필요 시 TensorFlow OD API로 fine-tune → tflite_convert --quantize → edgetpu_compiler.

---

## 7. 협업 규칙

1. 브랜치 : main(보호) / dev / feature/*
2. 커밋 규칙 : type(scope): subject 예) feat(tracker): add KCF re-init
3. PR 승인 : 2인 이상 리뷰 후 머지
4. CI : lint + pytest + gtest + build (GitHub Actions)
5. 대용량 파일 : models/ 가중치는 Git LFS 사용

---

## 8. 라이선스

- 코드와 노트북: MIT
- 책 내용 요약·인용: © O’Reilly Media – 공정 사용 범위 내 인용

---

Happy coding and committing! 🚀
