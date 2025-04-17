import os
import cv2
import numpy as np


def cargar_imagen(folder, image_name):

    print("Cargando imagen...")

    image_path = os.path.join(folder, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: No se pudo cargar la imagen en {image_path}")
        exit()
    else:
        cv2.imshow("Imagen original", image)  # Mostrar la imagen

    print(f"Tamaño de la imagen: {image.shape[1]}x{image.shape[0]} píxeles")

    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )  #   para convertir la imagen a escala de grises

    cv2.imshow("escala grises", gray)
    return image, image_name, gray


def ajustar_temperatura_calida(img, factor_rojo=1.2, factor_azul=0.8):
    b, g, r = cv2.split(img) # separamos canales R, G y B

    r = cv2.convertScaleAbs(r, alpha=factor_rojo, beta=0)  # aumentar rojo
    b = cv2.convertScaleAbs(b, alpha=factor_azul, beta=0)  # disminuir azul

    # volvemos a unir los canales
    img_calida = cv2.merge([b, g, r])
    return img_calida


def suaviza_iluminacion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    iluminacion = cv2.GaussianBlur(
        gray, (101, 101), 0
    )  # aplicamos desenfoque gaussiano en escala de grises
    iluminacion_bgr = cv2.cvtColor(iluminacion, cv2.COLOR_GRAY2BGR)

    img_float = img.astype(np.float32)  # convertimos a float32 para calculos
    iluminacion_float = iluminacion_bgr.astype(np.float32)

    img_corr = (img_float - iluminacion_float) + 128  # 128 es valor medio entre 0 y 255
    img_corr = np.clip(img_corr, 0, 255).astype(
        np.uint8
    )  # valores entre 0 y 255 y reconvertimos a uint8

    cv2.imshow("imagen tras suavizar iluminacion", img_corr)
    return img_corr


def fusion_suave(img1, img2, mask_gray, blur_size=21):  # antes blur_size=51
    mask_float = mask_gray.astype(np.float32) / 255.0  # normalizamos valores
    mask_blur = cv2.GaussianBlur(mask_float, (blur_size, blur_size), 0)
    mask_blur_3ch = cv2.merge(
        [mask_blur] * 3
    )  # para conseguir trasnsiciones suaves entre las imagenes
    #  valores de mascara entre 0 y 1

    img1_float = img1.astype(np.float32)
    img2_float = img2.astype(np.float32)

    blended = img1_float * mask_blur_3ch + img2_float * (1 - mask_blur_3ch)
    # las areas mas blancas (cercanas a 1) para dominio de imagen normal y las areas mas negras (cercanas a 0) para dominio de la sobreexpuesta
    return np.clip(blended, 0, 255).astype(np.uint8)


def main():
    print("Entrando al main...")

    folder = r"C:\Users\lucib\Desktop\lpro"
    normal, normal_name, normal_gray = cargar_imagen(folder, "normal2.jpg")
    sobreexpuesta, sobreexpuesta_name, sobreexpuesta_gray = cargar_imagen(
        folder, "sobreexpuesta2.jpg"
    )
    cv2.imshow("sobreexpuesta incial", sobreexpuesta)

    sobreexpuesta_calida = ajustar_temperatura_calida(
        sobreexpuesta, factor_rojo=1.15, factor_azul=0.8
    )
    cv2.imshow("sobreexpuesta calida", sobreexpuesta_calida)

    if normal.shape != sobreexpuesta_calida.shape:
        print("Redimensionando imágenes para que coincidan...")
        sobreexpuesta_calida = cv2.resize(
            sobreexpuesta_calida, (normal.shape[1], normal.shape[0])
        )

    threshold = 70  # antes 100
    mask = cv2.threshold(normal_gray, threshold, 255, cv2.THRESH_BINARY)[1]

    fusionada = fusion_suave(normal, sobreexpuesta_calida, mask)
    resultado_iluminacion = suaviza_iluminacion(fusionada)

    new_size = (1152, 648)
    resultado_redimensionado = cv2.resize(resultado_iluminacion, new_size)

    # guardamos en carpeta de salida
    carpeta_salida = os.path.join(folder, "salida")
    os.makedirs(carpeta_salida, exist_ok=True)

    ruta_guardado = os.path.join(carpeta_salida, "resultado_listo2.jpg")
    cv2.imwrite(ruta_guardado, resultado_redimensionado)

    print(f"Resultado guardado en: {ruta_guardado}")

    # Mostrar resultados
    cv2.imshow("Resultado Final", resultado_redimensionado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
