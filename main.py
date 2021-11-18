# Carregando as dependencias
import cv2 as cv

def deteccao(frame):
    # Detecção
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    # Percorrer todas as detecções
    for (classid, score, box) in zip(classes, scores, boxes):

        # Gerando uma cor para a classe
        color = COLORS[int(classid) % len(COLORS)]

        # Pegando o nome da classe pelo id e o seu score de acuracia
        label = f"{class_names[classid[0]]} : {score}"

        # Desenhando a box da detecção
        cv.rectangle(frame, box, color, 2)

        # Escrevendo o nome da classe em cima da box do objeto
        cv.putText(frame, label, (box[0], box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv.imshow("Detectando martelo e chave de fenda", frame)


# Quando o tipo de detecção for imagem é necessário informar o caminho da mesma
tipo_deteccao = input("Escolha o tipo de detecção.\n[i] para imagem ou [w] para webcam:")

if tipo_deteccao != "i" and tipo_deteccao != "w":
    exit()

# Cores das classes
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

# Carregando as classes
with open("obj.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Carregando os pesos da rede neural
net = cv.dnn.readNet("yolov4-90%-accuracy.weights", "yolov4-custom.cfg")


# Setando os parametros da rede neural
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

if tipo_deteccao == "w":
    # Capturando o vídeo
    cap = cv.VideoCapture(0)

    # Lendo os frames do video
    while True:
        # Captura do frame
        _, frame = cap.read()

        #Chamando o método de detecção
        deteccao(frame)

        # Espera da resposta
        if cv.waitKey(1) == 27:
            break

    # Liberação da camera e destroi todas as janelas
    cap.release()
elif tipo_deteccao == "i":
    #capturando a imagem e a redimensionando
    frame = cv.imread("img-test/img10.jpg")
    frame = cv.resize(frame, dsize=(416, 416))

    # Chamando o método de detecção
    deteccao(frame)

    # Espera da resposta
    cv.waitKey()

cv.destroyAllWindows()