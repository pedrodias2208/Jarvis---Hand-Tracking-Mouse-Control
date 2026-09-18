import cv2
import time
import mediapipe as mp
import mouse
import os
import numpy as np

# Configuração da resolução do monitor do seu notebook
largura_da_tela_pc = 1920
altura_da_tela_pc = 1080

# Histórico para fazer o mouse deslizar sem tremer (Média Móvel)
historico_posicoes_x = []
historico_posicoes_y = []
quantidade_de_frames_para_suavizar = 4  

# Trava de segurança para o clique não disparar igual uma metralhadora
ja_executou_o_clique = False

# Localiza o arquivo da Inteligência Artificial
pasta_do_projeto = os.path.dirname(os.path.abspath(__file__))
local_do_arquivo_da_ia = os.path.join(pasta_do_projeto, 'hand_landmarker.task')

# Configurações do MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

configuracoes_da_ia = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=local_do_arquivo_da_ia),
    running_mode=RunningMode.VIDEO, 
    num_hands=1,                    
    min_hand_detection_confidence=0.5, 
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

cerebro_da_ia = HandLandmarker.create_from_options(configuracoes_da_ia)

# Liga a webcam do notebook
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("====================================================")
print("     JARVIS V4 - NOVO CLIQUE NA BASE DO INDICADOR   ")
print(" Toque a ponta do dedão na base do indicador.       ")
print("====================================================")

with cerebro_da_ia:
    while True:
        leu_com_sucesso, imagem_da_camera = webcam.read()
        if not leu_com_sucesso:
            continue
            
        imagem_da_camera = cv2.flip(imagem_da_camera, 1)
        imagem_em_rgb = cv2.cvtColor(imagem_da_camera, cv2.COLOR_BGR2RGB)
        altura_da_imagem_camera, largura_da_imagem_camera, _ = imagem_da_camera.shape
        
        imagem_formato_ia = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagem_em_rgb)
        tempo_atual_em_milissegundos = int(time.time() * 1000)
        
        dados_da_mao_detectada = cerebro_da_ia.detect_for_video(imagem_formato_ia, tempo_atual_em_milissegundos)
        
        if dados_da_mao_detectada.hand_landmarks:
            for lista_de_juntas in dados_da_mao_detectada.hand_landmarks:
                
                # Coordenadas em pixels para o MOVIMENTO do mouse (Ponta do Indicador - Ponto 8)
                pixel_x_ponta_indicador = int(lista_de_juntas[8].x * largura_da_imagem_camera)
                pixel_y_ponta_indicador = int(lista_de_juntas[8].y * altura_da_imagem_camera)
                
                # Coordenadas em pixels para a NOVA LÓGICA DO CLIQUE
                # Ponto 4 = Ponta do Polegar (Dedão)
                pixel_x_ponta_polegar = int(lista_de_juntas[4].x * largura_da_imagem_camera)
                pixel_y_ponta_polegar = int(lista_de_juntas[4].y * altura_da_imagem_camera)
                
                # Ponto 5 = Base do Indicador (Onde o dedo começa na palma)
                pixel_x_base_indicador = int(lista_de_juntas[5].x * largura_da_imagem_camera)
                pixel_y_base_indicador = int(lista_de_juntas[5].y * altura_da_imagem_camera)
                
                # Calcula a distância matemática entre a PONTA DO DEDÃO e a BASE DO INDICADOR
                posicao_base_indicador_vetor = np.array([pixel_x_base_indicador, pixel_y_base_indicador])
                posicao_polegar_vetor = np.array([pixel_x_ponta_polegar, pixel_y_ponta_polegar])
                distancia_para_clique = np.linalg.norm(posicao_base_indicador_vetor - posicao_polegar_vetor)
                
                # ==============================================================
                # MOVIMENTO DO MOUSE (Sempre guiado pela ponta do indicador)
                # ==============================================================
                posicao_alvo_mouse_x = int(lista_de_juntas[8].x * largura_da_tela_pc)
                posicao_alvo_mouse_y = int(lista_de_juntas[8].y * altura_da_tela_pc)
                
                historico_posicoes_x.append(posicao_alvo_mouse_x)
                historico_posicoes_y.append(posicao_alvo_mouse_y)
                
                if len(historico_posicoes_x) > quantidade_de_frames_para_suavizar:
                    historico_posicoes_x.pop(0)
                    historico_posicoes_y.pop(0)
                
                posicao_final_mouse_x = sum(historico_posicoes_x) // len(historico_posicoes_x)
                posicao_final_mouse_y = sum(historico_posicoes_y) // len(historico_posicoes_y)
                
                mouse.move(posicao_final_mouse_x, posicao_final_mouse_y, absolute=True, duration=0)
                
                # ==============================================================
                # NOVA LÓGICA DO CLIQUE
                # ==============================================================
                # Como a base do indicador é mais próxima do dedão que a ponta, 
                # deixei o limite em 28 pixels para o clique ser bem firme.
                if distancia_para_clique < 28:
                    if not ja_executou_o_clique:
                        mouse.click('left')
                        ja_executou_o_clique = True
                        print("💥 CLIQUE NA BASE!                          ", end="\r")
                    
                    # Desenha linha vermelha unindo o dedão com a base do indicador
                    cv2.line(imagem_da_camera, (pixel_x_ponta_polegar, pixel_y_ponta_polegar), 
                             (pixel_x_base_indicador, pixel_y_base_indicador), (0, 0, 255), 2)
                else:
                    if distancia_para_clique > 38:
                        ja_executou_o_clique = False
                    
                    # Desenha linha verde se estiverem afastados
                    cv2.line(imagem_da_camera, (pixel_x_ponta_polegar, pixel_y_ponta_polegar), 
                             (pixel_x_base_indicador, pixel_y_base_indicador), (0, 255, 0), 1)
                
                # Desenha os círculos guias
                cv2.circle(imagem_da_camera, (pixel_x_ponta_indicador, pixel_y_ponta_indicador), 6, (0, 255, 255), -1) # Guia do Mouse (Amarelo)
                cv2.circle(imagem_da_camera, (pixel_x_ponta_polegar, pixel_y_ponta_polegar), 6, (255, 0, 0), -1)       # Dedão (Azul)
                cv2.circle(imagem_da_camera, (pixel_x_base_indicador, pixel_y_base_indicador), 6, (0, 128, 255), -1)   # Base Indicador (Laranja)
                
        cv2.imshow("Jarvis - Clique na Base", imagem_da_camera)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

webcam.release()
cv2.destroyAllWindows()