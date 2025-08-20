import cv2
import pytesseract
import os
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15, 5
    )

    return thresh

# ---------- OCR ----------
def extract_text(image):
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, lang='por', config=custom_config)
    return text.strip()

# ---------- CAPTURA DA CÂMERA ----------
def capture_and_scan():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        return

    print("Pressione 'ESPACO' para capturar ou 'ESC' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar imagem.")
            break

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 32:  # ESPACO
            print("Capturando imagem...")

            processed_img = preprocess_image(frame)

            # Salva para debug
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("capturas", exist_ok=True)
            cv2.imwrite(f"capturas/original_{timestamp}.png", frame)
            cv2.imwrite(f"capturas/processada_{timestamp}.png", processed_img)

            # Mostra a imagem processada
            cv2.imshow("Imagem Processada", processed_img)

            # Extrai texto
            ocr_text = extract_text(processed_img)
            print("\n===== TEXTO DETECTADO =====")
            print(ocr_text if ocr_text else "[Nenhum texto detectado]")
            print("===========================\n")

    cap.release()
    cv2.destroyAllWindows()

# ---------- EXECUÇÃO ----------
if __name__ == "__main__":
    capture_and_scan()
