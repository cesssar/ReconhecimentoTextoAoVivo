import cv2 as cv
import pytesseract as ocr

camera = cv.VideoCapture(0)
rodando = True
img_contador = 0

print('Iniciando camera...')

def tratamento_img(frame):
    # converte a imagem para gray scale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # tratamento da imagem
    ret, thresh1 = cv.threshold(gray, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)
    rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (18,18))
    dilation = cv.dilate(thresh1, rect_kernel, iterations=1)
    contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # copia da imagem
    im2 = frame.copy()
    return contours, im2

def extracao_texto(contours, im2):
    # laco identificar contornos
    # cada parte retangular é passado para identificar texto e extrair
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        rect = cv.rectangle(im2, (x,y),(x + w, y + h), (0,255,0), 2)
        cropped = im2[y:y +h, x:x + w]
        text = ocr.image_to_string(cropped, lang='por')
        text = str(text).replace('\n', ' ').replace('\t', ' ')
        if len(text) > 0:
            print(f'Texto detectado: {text}')


while rodando:
    status, frame = camera.read()
    print('Imagem capturada...')
    if not status: #caso de erro
        rodando = False

    contours, im2 = tratamento_img(frame)
    extracao_texto(contours, im2)
    