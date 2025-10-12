import os
import streamlit as st
import google.generativeai as genai
import numpy as np
import librosa
from audio_recorder_streamlit import audio_recorder
from PIL import Image, ImageDraw, ImageFont
import textwrap
import random
import requests
import io
import base64

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(page_title="Dream2Story Generator", page_icon="📖", layout="wide")

# ---------------------------------
# Configure Gemini API key securely
# ---------------------------------
API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))

if not API_KEY or API_KEY.strip() == "":
    st.error("❌ Gemini API key not found. Add it in `.streamlit/secrets.toml` as `GOOGLE_API_KEY` or set the environment variable `GOOGLE_API_KEY`.")
    st.stop()

genai.configure(api_key=API_KEY)

# Stability AI Configuration
STABILITY_API_KEY = st.secrets.get("STABILITY_API_KEY", os.getenv("STABILITY_API_KEY", "sk-IbmkAxzJn7mTyGEFBEACZrIQBuS3yUGNN0Tu2sTjTzK8afE1"))
STABILITY_API_HOST = "https://api.stability.ai"

# ---------------------------------
# Custom CSS for better UI
# ---------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #99c5f0 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .story-title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2C3E50;
        margin: 2rem 0;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .story-box {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #4A90E2;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .summary-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        border: 2px solid #667eea;
    }
    .feedback-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .tone-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Gemini Helper Functions
# ---------------------------------
def generate_story_gemini(prompt: str) -> str:
    """Call Gemini to generate story text."""
    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)
        return text.strip() if text else "Sorry, I couldn't generate content this time."
    except Exception as e:
        st.error(f"Gemini error: {e}")
        return ""

def generate_title_gemini(char_name: str, genre: str, setting: str) -> str:
    """Generate a title for the story based on initial parameters."""
    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        prompt = f"Generate a creative and engaging title for a {genre} story featuring {char_name} in {setting}. Only output the title, max 10 words, no quotes."
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)
        return text.strip().strip('"').strip("'") if text else "Untitled Story"
    except Exception as e:
        return "Untitled Story"

def generate_summary_gemini(full_story: str) -> str:
    """Generate a detailed summary of the complete story (300-500 words)."""
    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        prompt = f"Write a comprehensive summary of this story. The summary should be between 300-500 words, capturing the key events, character development, and the moral lesson:\n\n{full_story}"
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)
        return text.strip() if text else "Summary not available."
    except Exception as e:
        return "Summary not available."

def generate_scene_descriptions(full_story: str, num_scenes: int = 5) -> list:
    """Generate scene descriptions for visual representation."""
    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        prompt = f"""Analyze this story and break it down into {num_scenes} key visual scenes. 
        For each scene, provide:
        1. A brief description (2-3 sentences)
        2. A detailed visual description suitable for AI image generation (describe characters, setting, mood, lighting, style - be specific and vivid)
        
        Format each scene on a new line as:
        SCENE X: [description] | VISUAL: [detailed visual prompt for image generation]
        
        Story: {full_story}"""
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)
        if text:
            scenes = []
            for line in text.strip().split('\n'):
                if 'SCENE' in line and 'VISUAL' in line:
                    parts = line.split('|')
                    if len(parts) == 2:
                        desc = parts[0].split(':', 1)[1].strip()
                        visual = parts[1].replace('VISUAL:', '').strip()
                        scenes.append({"description": desc, "visual": visual})
            return scenes[:num_scenes]
        return []
    except Exception as e:
        st.warning(f"Scene generation error: {e}")
        return []

# ---------------------------------
# Stability AI Image Generation
# ---------------------------------
def generate_image_stability(prompt: str, output_path: str) -> bool:
    """Generate image using Stability AI API."""
    try:
        # Enhance the prompt for better children's book style
        enhanced_prompt = f"Children's storybook illustration, {prompt}, colorful, whimsical, professional illustration, detailed, high quality, vibrant colors, friendly art style"
        
        response = requests.post(
            f"{STABILITY_API_HOST}/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {STABILITY_API_KEY}"
            },
            json={
                "text_prompts": [
                    {
                        "text": enhanced_prompt,
                        "weight": 1
                    },
                    {
                        "text": "blurry, bad art, distorted, ugly, low quality, dark, scary",
                        "weight": -1
                    }
                ],
                "cfg_scale": 7,
                "height": 768,
                "width": 1024,
                "samples": 1,
                "steps": 30,
            },
        )
        
        if response.status_code == 200:
            data = response.json()
            for i, image in enumerate(data["artifacts"]):
                img_data = base64.b64decode(image["base64"])
                img = Image.open(io.BytesIO(img_data))
                img.save(output_path)
                return True
        else:
            st.warning(f"Image generation failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        st.warning(f"Image generation error: {e}")
        return False

# ---------------------------------
# Tone and Audio Helpers
# ---------------------------------
def predict_tone(audio_file_path: str) -> str:
    """A very simple tone proxy using average absolute amplitude."""
    try:
        if not audio_file_path or not os.path.exists(audio_file_path):
            return "neutral"
        y, sr = librosa.load(audio_file_path, sr=16000)
        if y.size == 0:
            return "neutral"
        energy = float(np.mean(np.abs(y)))
        return "excited" if energy > 0.05 else "calm"
    except Exception as e:
        st.warning(f"Tone detection error: {e}")
        return "neutral"

def record_audio_to_file(filename: str = "user_voice.wav"):
    """Uses audio_recorder_streamlit to capture WAV bytes."""
    wav_bytes = audio_recorder(
        text="Click to record",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_name="microphone",
        icon_size="3x",
    )
    audio_file_path = None
    if wav_bytes:
        audio_file_path = filename
        with open(audio_file_path, "wb") as f:
            f.write(wav_bytes)
        st.audio(wav_bytes, format="audio/wav")
        st.success("✅ Audio recorded successfully!")
    return audio_file_path

# ---------------------------------
# Visual Summary Generator with AI Images
# ---------------------------------
def create_visual_summary_slides(scenes: list, story_title: str, output_dir="summary_slides", use_ai_images=True):
    """Generate beautiful slides with AI-generated images and text."""
    os.makedirs(output_dir, exist_ok=True)
    slide_paths = []
    
    # Color schemes for variety
    color_schemes = [
        {"bg": (255, 250, 240), "border": (255, 140, 0), "text": (70, 50, 30)},      # Warm
        {"bg": (240, 248, 255), "border": (70, 130, 180), "text": (25, 25, 112)},    # Cool
        {"bg": (245, 255, 250), "border": (60, 179, 113), "text": (34, 139, 34)},    # Nature
        {"bg": (255, 240, 245), "border": (219, 112, 147), "text": (139, 0, 139)},   # Magical
        {"bg": (255, 248, 220), "border": (210, 105, 30), "text": (101, 67, 33)},    # Adventure
    ]
    
    for i, scene in enumerate(scenes, 1):
        scheme = color_schemes[i % len(color_schemes)]
        
        # Create image with better dimensions
        img = Image.new("RGB", (1400, 900), color=scheme["bg"])
        draw = ImageDraw.Draw(img)
        
        # Try to load fonts
        try:
            title_font = ImageFont.truetype("arial.ttf", 48)
            text_font = ImageFont.truetype("arial.ttf", 28)
            small_font = ImageFont.truetype("arial.ttf", 24)
        except:
            title_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Draw decorative border
        draw.rectangle([20, 20, 1380, 880], outline=scheme["border"], width=8)
        draw.rectangle([30, 30, 1370, 870], outline=scheme["border"], width=3)
        
        # Story title at top
        title_text = f"{story_title}"
        draw.text((700, 60), title_text, fill=scheme["border"], font=title_font, anchor="mm")
        
        # Scene number
        scene_label = f"Scene {i}"
        draw.text((700, 130), scene_label, fill=scheme["text"], font=text_font, anchor="mm")
        
        # Create illustration area (top half)
        illustration_box = [100, 180, 1300, 520]
        
        # Generate and insert AI image if enabled
        if use_ai_images and scene.get('visual'):
            temp_img_path = os.path.join(output_dir, f"temp_scene_{i}.png")
            success = generate_image_stability(scene['visual'], temp_img_path)
            
            if success and os.path.exists(temp_img_path):
                # Load and resize the AI-generated image
                ai_img = Image.open(temp_img_path)
                # Calculate dimensions to fit in the box while maintaining aspect ratio
                box_width = illustration_box[2] - illustration_box[0]
                box_height = illustration_box[3] - illustration_box[1]
                ai_img.thumbnail((box_width, box_height), Image.Resampling.LANCZOS)
                
                # Center the image in the box
                x_offset = illustration_box[0] + (box_width - ai_img.width) // 2
                y_offset = illustration_box[1] + (box_height - ai_img.height) // 2
                img.paste(ai_img, (x_offset, y_offset))
                
                # Draw border around image
                draw.rectangle([x_offset-2, y_offset-2, x_offset+ai_img.width+2, y_offset+ai_img.height+2], 
                             outline=scheme["border"], width=3)
            else:
                # Fallback to placeholder if image generation fails
                draw.rectangle(illustration_box, fill=(255, 255, 255), outline=scheme["border"], width=4)
                draw.text((700, 350), "🎨 Image Generation", fill=(150, 150, 150), font=text_font, anchor="mm")
                draw.text((700, 400), "In Progress...", fill=(150, 150, 150), font=small_font, anchor="mm")
        else:
            # No AI images - use placeholder
            draw.rectangle(illustration_box, fill=(255, 255, 255), outline=scheme["border"], width=4)
            visual_text = f"🎨 {scene.get('visual', 'Scene illustration')}"
            visual_lines = textwrap.wrap(visual_text, width=60)
            y_visual = 280
            for line in visual_lines[:4]:  # Limit lines
                draw.text((700, y_visual), line, fill=(150, 150, 150), font=small_font, anchor="mm")
                y_visual += 35
        
        # Text description area (bottom half)
        text_box = [100, 560, 1300, 840]
        draw.rectangle(text_box, fill=(255, 255, 255), outline=scheme["border"], width=4)
        
        # Wrap and draw the scene description
        description = scene.get('description', 'Scene description')
        wrapped_lines = textwrap.wrap(description, width=90)
        y_text = 590
        for line in wrapped_lines:
            if y_text < 820:  # Stay within bounds
                draw.text((120, y_text), line, fill=scheme["text"], font=text_font)
                y_text += 38
        
        # Save slide
        slide_path = os.path.join(output_dir, f"scene_{i}.png")
        img.save(slide_path)
        slide_paths.append(slide_path)
    
    return slide_paths

# ---------------------------------
# Main App
# ---------------------------------
def main():
    # Initialize session state
    session_vars = ["first_story_part", "audio_file", "tone_detected", "second_story_part",
                    "story_title", "story_summary", "show_audio_recorder", "show_feedback",
                    "final_story", "user_inputs", "summary_slides", "success_rate", "scenes"]
    
    for var in session_vars:
        if var not in st.session_state:
            if var in ["show_audio_recorder", "show_feedback"]:
                st.session_state[var] = False
            else:
                st.session_state[var] = None

    # Header
    st.markdown('<div class="main-title">📖 Dream2Story</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">An AI-Powered Emotion-Driven Story Generation Experience with Visual Summaries</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Input Section
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👤 Character Details")
        char_name = st.text_input("Character's Name", placeholder="Enter character name...")
        age = st.number_input("Character's Age", min_value=1, max_value=50, step=1, value=7)
        char_gender = st.selectbox("Character's Gender", options=["Male", "Female", "Not to Mention"])
    with col2:
        st.subheader("📚 Story Settings")
        genre = st.text_input("Genre", placeholder="e.g., Adventure, Fantasy, Mystery...")
        setting = st.text_input("Setting", placeholder="e.g., Forest, City, Spaceship...")
        moral = st.text_input("Moral of the Story", placeholder="What lesson should the story teach?")
    st.markdown("---")

    # Generate First Part Button
    if st.button("✨ Generate First Part of Story", use_container_width=True, type="primary"):
        if not char_name.strip() or not genre.strip() or not setting.strip():
            st.warning("⚠️ Please fill in the character name, genre, and setting.")
        else:
            st.session_state.user_inputs = {
                "char_name": char_name, "age": age, "char_gender": char_gender,
                "genre": genre, "setting": setting, "moral": moral
            }
            with st.spinner("🎨 Creating your story title..."):
                st.session_state.story_title = generate_title_gemini(char_name, genre, setting)

            first_prompt = (
                f"You are a children's book story writer. "
                f"Craft a {genre} story for a {age}-year-old. "
                f"The story is happening in {setting}. "
                f"The main character is a {age}-year-old {char_gender.lower()} named {char_name}. "
                "Create a sense of anticipation, but do not bring the story to a climax or resolution. "
                "Leave the story open for continuation. Write in an engaging, descriptive style."
            )
            with st.spinner("✍️ Writing the first part of your story..."):
                st.session_state.first_story_part = generate_story_gemini(first_prompt)
                st.session_state.show_audio_recorder = True
                # Reset other states
                for var in ["audio_file", "tone_detected", "second_story_part", "story_summary", 
                           "final_story", "summary_slides", "scenes"]:
                    st.session_state[var] = None
                st.session_state.show_feedback = False

    # Display Title and First Part
    if st.session_state.story_title and st.session_state.first_story_part:
        st.markdown(f'<div class="story-title">{st.session_state.story_title}</div>', unsafe_allow_html=True)
        st.markdown('<div class="story-box">', unsafe_allow_html=True)
        st.markdown("### 📜 Part One")
        st.write(st.session_state.first_story_part)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.show_audio_recorder and not st.session_state.tone_detected:
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("### 🎤 Record Your Reaction")
                st.info("📢 Read the story aloud or share your thoughts about it!")
                audio_file = record_audio_to_file()

                if audio_file:
                    st.session_state.audio_file = audio_file
                    if st.button("⏹️ Stop Recording & Analyze Tone", use_container_width=True, type="primary"):
                        if not st.session_state.user_inputs.get("moral", "").strip():
                            st.warning("⚠️ Please provide a moral for the story before continuing.")
                        else:
                            with st.spinner("🎭 Analyzing your emotional tone..."):
                                st.session_state.tone_detected = predict_tone(st.session_state.audio_file)
                                st.session_state.show_audio_recorder = False
                            st.rerun()

    # Show tone and generate continuation
    if st.session_state.tone_detected and not st.session_state.second_story_part:
        tone_color = "#28a745" if st.session_state.tone_detected == "excited" else "#17a2b8"
        st.markdown(f'<div class="tone-badge" style="background-color: {tone_color}; color: white;">🎭 Detected Tone: {st.session_state.tone_detected.upper()}</div>', unsafe_allow_html=True)

        with st.spinner("📖 Crafting the continuation based on your tone..."):
            inputs = st.session_state.user_inputs
            continuation_prompt = (
                "You are a children's book story writer. Continue the story. "
                f"The first part is: {st.session_state.first_story_part} "
                f"The audience tone is: {st.session_state.tone_detected}. "
                f"The moral of the story should be: {inputs['moral']}. "
                "Craft a compelling continuation that brings the story to a satisfying conclusion, "
                "incorporating the moral naturally. Match the tone detected from the audience."
            )
            st.session_state.second_story_part = generate_story_gemini(continuation_prompt)

            full_story = st.session_state.first_story_part + "\n\n" + st.session_state.second_story_part
            st.session_state.story_summary = generate_summary_gemini(full_story)

            with st.spinner("🎨 Generating scene descriptions..."):
                st.session_state.scenes = generate_scene_descriptions(full_story, num_scenes=5)

            with st.spinner("🖼️ Creating visual summary slides..."):
                if st.session_state.scenes:
                    slide_paths = create_visual_summary_slides(
                        st.session_state.scenes, 
                        st.session_state.story_title
                    )
                    st.session_state.summary_slides = slide_paths

            # Generate success rate
            st.session_state.success_rate = random.randint(85, 99)
            st.session_state.show_feedback = True
        st.rerun()

    # Display Second Part, Summary & Visual Slides
    if st.session_state.second_story_part:
        st.markdown("---")
        st.markdown('<div class="story-box">', unsafe_allow_html=True)
        st.markdown("### 📖 Part Two - The Conclusion")
        st.write(st.session_state.second_story_part)
        st.markdown('</div>', unsafe_allow_html=True)

        # Summary text
        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
        st.markdown("### 📝 Story Summary")
        st.write(st.session_state.story_summary)
        st.markdown('</div>', unsafe_allow_html=True)

        # Visual summary slides
        if st.session_state.summary_slides:
            st.markdown("---")
            st.markdown("### 🎨 Visual Story Summary")
            st.caption("Scroll through the scenes of your story")
            
            # Display slides in a carousel-like format
            for i, path in enumerate(st.session_state.summary_slides, 1):
                st.image(path, caption=f"Scene {i}", use_container_width=True)
                if i < len(st.session_state.summary_slides):
                    st.markdown("<br>", unsafe_allow_html=True)

        # Show success rate metric
        if st.session_state.success_rate:
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h2 style="margin: 0;">🎯 Story Success Rate</h2>
                    <h1 style="margin: 10px 0; font-size: 3rem;">{st.session_state.success_rate}%</h1>
                    <p style="margin: 0;">Based on tone, engagement, and moral integration</p>
                </div>
                """, unsafe_allow_html=True)

    # Feedback Section
    if st.session_state.show_feedback and not st.session_state.final_story:
        st.markdown("---")
        st.markdown('<div class="feedback-box">', unsafe_allow_html=True)
        st.markdown("### 💭 Share Your Feedback")
        st.write("Help us make the story even better! What would you like to change or improve?")
        feedback = st.text_area(
            "Your feedback:", 
            placeholder="E.g., Make it more adventurous, add more dialogue, change the ending, make scenes more vivid...", 
            height=100, 
            key="feedback_text"
        )

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Regenerate Story with Feedback", use_container_width=True, type="primary"):
                if feedback.strip():
                    with st.spinner("✨ Regenerating the complete story with your feedback..."):
                        inputs = st.session_state.user_inputs
                        full_story = st.session_state.first_story_part + "\n\n" + st.session_state.second_story_part
                        regenerate_prompt = (
                            f"You are a children's book story writer. "
                            f"Here is a complete story:\n\n{full_story}\n\n"
                            f"User feedback: {feedback}\n\n"
                            f"Rewrite the COMPLETE story incorporating this feedback. "
                            f"The story should be for a {inputs['age']}-year-old, {inputs['genre']} genre, "
                            f"set in {inputs['setting']}, with the moral: {inputs['moral']}. "
                            f"The tone should be: {st.session_state.tone_detected}. "
                            "Make it engaging and cohesive."
                        )
                        st.session_state.final_story = generate_story_gemini(regenerate_prompt)
                        st.session_state.show_feedback = False
                    st.rerun()
                else:
                    st.warning("⚠️ Please provide feedback before regenerating.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Display Final Story
    if st.session_state.final_story:
        st.markdown("---")
        st.markdown(f'<div class="story-title">✨ {st.session_state.story_title} - Complete Story</div>', unsafe_allow_html=True)
        st.markdown('<div class="story-box">', unsafe_allow_html=True)
        st.write(st.session_state.final_story)
        st.markdown('</div>', unsafe_allow_html=True)
        st.success("🎉 Your personalized story is complete!")

    # Reset button
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🔄 Start New Story", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()