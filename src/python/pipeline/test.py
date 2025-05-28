import cv2
import numpy as np

def pseudo_ir(frame: np.ndarray) -> np.ndarray:
    """
    입력 BGR 프레임을 pseudo-IR (thermal) 스타일로 변환합니다.
    """
    # 1) BGR → Gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 2) CLAHE 적용 (contrast enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray)
    # 3) Gray → BGR (다른 코드와 통일시키기 위함)
    frame = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)

    return frame

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    cv2.namedWindow("Original | Pseudo-IR", cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ir = pseudo_ir(frame)
        # 원본(BGR)과 IR(BGR) 이미지를 가로로 붙이기
        combined = np.hstack((frame, ir))

        cv2.imshow("Original | Pseudo-IR", combined)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
