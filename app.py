import streamlit as st
import os, json, joblib
from openai import OpenAI

st.set_page_config(page_title='Clasificación de Correos Spam', layout='centered')
st.title('Detección de Correos Spam - Eduin Fabricio Zelaya')

@st.cache_resource
def cargar_modelo():
    # Rutas correspondientes a los archivos guardados en el entrenamiento
    pkl_modelos = 'modelos_clasificacion_spam.pkl'
    meta = 'model_metadata.json'
    
    modelos = joblib.load(pkl_modelos)
    with open(meta, encoding='utf-8') as f:
        metadata = json.load(f)
    
    return modelos, metadata

@st.cache_data
def cargar_base():
    csv = 'spam_espanol.csv'
    if os.path.exists(csv):
        return pd.read_csv(csv)
    return None

try:
    modelos, metadata = cargar_modelo()
    df = cargar_base()
    modelo_seleccionado_defecto = metadata.get('mejor_modelo_cv', 'SVM (con escalado)')
except Exception as e:
    st.error(f"Error al cargar los recursos: {e}")
    st.stop()

st.caption(metadata.get('nombre_modelo', 'Clasificador de Spam / Ham'))

# Formulario para ingresar nuevos mensajes de correo
with st.form('datos_correo'):
    st.subheader('Análisis de Mensaje Nuevo')
    
    # Entrada de texto para el correo
    correo_texto = st.text_area('Contenido del Correo', placeholder='Pegue el texto del correo aquí para evaluar...', height=150)
    
    # Selección del modelo a usar para la predicción
    algoritmo = st.selectbox('Seleccione el Algoritmo de Predicción', list(modelos.keys()), index=list(modelos.keys()).index(modelo_seleccionado_defecto))
    
    enviar = st.form_submit_button('Evaluar Mensaje')

if enviar:
    if not correo_texto.strip():
        st.warning("Por favor, ingrese un texto válido para evaluar.")
    else:
        # Obtener la predicción del modelo seleccionado
        modelo_actuante = modelos[algoritmo]
        prediccion = modelo_actuante.predict([correo_texto])[0]
        
        resultado_clase = "Spam (Peligroso / Bloquear)" if prediccion == 'spam' else "Normal (Seguro)"
        color_alerta = "red" if prediccion == 'spam' else "green"
        
        st.markdown(f"### Diagnóstico: :{color_alerta}[{resultado_clase}]")
        st.write(f"Algoritmo utilizado: **{algoritmo}**")
        
        # Integración opcional con Groq API / OpenAI para dar explicación experta
        api_key = st.secrets.get('GROQ_API_KEY', os.getenv('GROQ_API_KEY', ''))
        
        if api_key:
            prompt = f'''
            Actúa como un experto en Seguridad Informática y Especialista en Anti-Phishing.
            
            Explica brevemente por qué un correo con el siguiente contenido ha sido catalogado como {resultado_clase}.
            Proporciona 3 recomendaciones de seguridad inmediatas para el usuario.
            
            Contenido del correo:
            "{correo_texto}"
            
            Resultado del modelo:
            - Clasificación: {resultado_clase}
            - Modelo actuante: {algoritmo}
            '''
            
            try:
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1"
                )
                
                respuesta = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {
                            "role": "system",
                            "content": "Eres un asesor experto en seguridad de la información y analista de ingeniería social."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                
                texto = respuesta.choices[0].message.content
                st.info(texto)
                
            except Exception as e:
                st.warning(f'Error al conectar con el servicio de IA: {e}')
        else:
            st.info('Para obtener recomendaciones automatizadas de seguridad mediante IA, configure su GROQ_API_KEY en los secretos de Streamlit.')

if df is not None:
    st.divider()
    st.write('Historial Reciente de Correos Procesados (Dataset de Entrenamiento)')
    st.dataframe(df.head(15), use_container_width=True)
