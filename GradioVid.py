import os
import gradio as gr
from moviepy.editor import *
from moviepy.config import change_settings

change_settings({"IMAGEMAGICK_BINARY": "C:/Program Files/ImageMagick/magick.exe"})

def generer_video(topic, age_min, age_max, niveau, creativite, humour, contexte):
    # Ensure the directory exists
    output_dir = './videos'
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure the video filename is valid
    video_filename = f"{topic.replace(' ', '_').replace('\'', '')}.mp4"
    video_path = os.path.join(output_dir, video_filename)

    # Create text clip with user input
    age_cible = f"{age_min} à {age_max} ans"
    txt_clip = TextClip(
        f"Sujet: {topic}\nÂge cible: {age_cible}\nNiveau: {niveau}\nCréativité: {creativite}\nHumour: {humour}\nContexte: {contexte}",
        fontsize=30, color='white', bg_color='black'
    )
    txt_clip = txt_clip.set_position(('center', 'center')).set_duration(10)  # Position and duration of the text

    # Create the final video clip
    final_clip = CompositeVideoClip([txt_clip], size=(640, 480))

    try:
        # Write the video file
        final_clip.write_videofile(video_path, codec='libx264', fps=24)
    except Exception as e:
        return f"Error: {str(e)}"
    
    return video_path

def creer_interface():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Générateur de Vidéo Personnalisée
            **Veuillez remplir les informations ci-dessous pour générer une vidéo personnalisée.**
            """
        )

        with gr.Row():
            with gr.Column():
                topic = gr.Textbox(label="Sujet", placeholder="Quel est le sujet de la vidéo ?")
                age_min = gr.Dropdown(choices=[str(i) for i in range(3, 19)], label="Âge Minimum")
                age_max = gr.Dropdown(choices=[str(i) for i in range(4, 20)], label="Âge Maximum")

            with gr.Column():
                niveau = gr.Dropdown(
                    choices=["Niveau débutant", "Niveau intermédiaire", "Niveau avancé"],
                    label="Description du Niveau"
                )
                creativite = gr.Radio(choices=["Haute", "Modérée", "Faible"], label="Note de Créativité")
                humour = gr.Radio(choices=["Forte", "Légère", "Aucun"], label="Note d'Humour")
                contexte = gr.Textbox(label="Contexte",
                                      placeholder="Ajoutez des informations qui peuvent influencer la vidéo")

        bouton = gr.Button("Générer Vidéo", variant="primary")

        sortie = gr.Video(label="Vidéo Générée")

        bouton.click(generer_video, inputs=[topic, age_min, age_max, niveau, creativite, humour, contexte], outputs=sortie)

    return demo

if __name__ == "__main__":
    interface = creer_interface()
    interface.launch()
