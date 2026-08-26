import streamlit as st
import os
import json
import time
import asyncio
from dotenv import load_dotenv

# Load environment
load_dotenv('.env', override=True)

from tools.video_tools import (
    extract_youtube_chunks_api,
    get_video_id,
    get_video_duration
)
from tools.history_manager import (
    load_run_history,
    save_run_to_history,
    clear_run_history
)

st.set_page_config(
    page_title="Video Intelligence Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme-safe, minimalist styling
st.markdown("""
<style>
    .header-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .header-subtitle {
        font-size: 0.95rem;
        opacity: 0.75;
        margin-bottom: 1.2rem;
    }
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Sidebar -----------------
with st.sidebar:
    st.markdown("### ⚡ Video Studio")
    
    sample_videos = {
        "Nvidia Q4 Earnings Call (11m)": "https://www.youtube.com/watch?v=3MwxX1ee_gI",
        "Custom URL": ""
    }
    
    selected_sample = st.selectbox("Select Video Preset", list(sample_videos.keys()))
    
    if selected_sample == "Custom URL":
        default_url = ""
    else:
        default_url = sample_videos[selected_sample]
        
    video_url = st.text_input("YouTube URL", value=default_url, placeholder="https://www.youtube.com/watch?v=...")
    
    template = st.selectbox(
        "Intelligence Template",
        options=["financial", "insights", "chapters", "transcript"],
        index=0,
        format_func=lambda x: {
            "financial": "💼 Financial & Earnings Intelligence",
            "insights": "💡 Key Takeaways & Evidence",
            "chapters": "📑 Video Chapters & Bookmarks",
            "transcript": "📝 Line-by-Line Transcript"
        }[x]
    )
    
    run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    with st.expander("⚙️ Advanced Engine Settings", expanded=False):
        model = st.selectbox(
            "Model",
            options=["gemini-3.7-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
            index=0,
            help="Gemini 3.7 Flash operates in the global region."
        )
        chunk_size = st.number_input("Chunk Interval (seconds)", min_value=60, max_value=1200, value=350, step=30)
        max_concurrency = st.number_input("Max Concurrency", min_value=1, max_value=10, value=5, step=1)

    st.markdown("---")
    
    # Run History Accordion in Sidebar
    history_runs = load_run_history()
    with st.expander(f"📜 Run History ({len(history_runs)})", expanded=False):
        if history_runs:
            for idx, r in enumerate(history_runs[:8]):
                ts = r.get("timestamp", "").split(" ")[-1]
                v_id = r.get("video_id") or "Video"
                tpl = r.get("template", "run")
                dur = f"{r.get('elapsed_seconds')}s"
                btn_label = f"#{idx+1} {v_id} ({tpl}) - {dur}"
                if st.button(btn_label, key=f"hist_btn_{r.get('id', idx)}", use_container_width=True):
                    st.session_state["latest_result"] = r.get("report_markdown")
                    st.session_state["latest_url"] = r.get("video_url")
                    st.session_state["latest_template"] = r.get("template")
                    st.session_state["latest_elapsed"] = r.get("elapsed_seconds")
                    st.session_state["latest_model"] = r.get("model")
                    st.rerun()
                    
            if st.button("🗑️ Clear History", key="clear_hist_btn", use_container_width=True):
                clear_run_history()
                st.rerun()
        else:
            st.caption("No past runs recorded yet.")

# ----------------- Main Layout -----------------
st.markdown('<div class="header-title">⚡ Multimodal Video Intelligence Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Parallel asynchronous video intelligence powered by Google Gemini 3.7 Flash & Vertex AI</div>', unsafe_allow_html=True)

if not video_url and "latest_result" not in st.session_state:
    st.info("👈 Enter a YouTube URL or select a preset in the sidebar to begin.")
    st.stop()

display_url = video_url or st.session_state.get("latest_url", "")
video_id = get_video_id(display_url)

col_video, col_report = st.columns([1, 1.3], gap="medium")

# Left Column: Video & Metadata
with col_video:
    st.markdown("#### 📺 Video Source")
    if display_url:
        st.video(display_url)
        
    duration = get_video_duration(display_url)
    
    with st.container(border=True):
        st.markdown("**Video Summary**")
        m_c1, m_c2, m_c3 = st.columns(3)
        with m_c1:
            if duration:
                mins, secs = divmod(duration, 60)
                st.metric("Duration", f"{mins}m {secs}s")
            else:
                st.metric("Duration", "Auto")
        with m_c2:
            if duration:
                num_chunks = (duration + chunk_size - 1) // chunk_size
                st.metric("Chunks", f"{num_chunks} parallel")
            else:
                st.metric("Chunks", "1 stream")
        with m_c3:
            st.metric("Model", "Gemini 3.7")

# Right Column: Unified Intelligence Report
with col_report:
    st.markdown("#### 📊 Intelligence Output")
    
    if run_button and video_url:
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Initializing parallel chunk analysis...")
        
        def update_progress(chunk_idx, total_chunks, msg):
            pct = min(1.0, chunk_idx / total_chunks)
            progress_bar.progress(pct)
            status_text.text(msg)
            
        start_time = time.time()
        
        with st.spinner("Processing video stream with Gemini 3.7 Flash..."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result_markdown = loop.run_until_complete(
                    extract_youtube_chunks_api(
                        video_url=video_url,
                        model=model,
                        template=template,
                        chunk_size=chunk_size,
                        max_concurrency=max_concurrency,
                        progress_callback=update_progress
                    )
                )
                elapsed = time.time() - start_time
                progress_bar.progress(1.0)
                status_text.success(f"✅ Analysis completed in {elapsed:.1f}s")
                
                # Save to run history
                filename_md = f"outputs/video_{video_id}_{template}.md"
                with open(filename_md, "w", encoding="utf-8") as f_out:
                    f_out.write(result_markdown)
                    
                save_run_to_history(
                    video_url=video_url,
                    video_id=video_id,
                    template=template,
                    model=model,
                    duration_seconds=duration,
                    elapsed_seconds=elapsed,
                    report_markdown=result_markdown,
                    output_file=filename_md
                )
                
                # Cache in session state
                st.session_state["latest_result"] = result_markdown
                st.session_state["latest_url"] = video_url
                st.session_state["latest_template"] = template
                st.session_state["latest_elapsed"] = elapsed
                st.session_state["latest_model"] = model
            except Exception as e:
                status_text.error(f"❌ Extraction Error: {e}")
                st.stop()

    if "latest_result" in st.session_state:
        result_md = st.session_state["latest_result"]
        current_tpl = st.session_state.get("latest_template", template)
        elapsed_sec = st.session_state.get("latest_elapsed", 0.0)
        
        # Display clean badges above tabs
        b_col1, b_col2 = st.columns([1, 1])
        with b_col1:
            st.caption(f"Template: **{current_tpl.upper()}** | Processed in **{elapsed_sec}s**")
        
        tab_report, tab_raw, tab_export = st.tabs(["📑 Unified Report", "📝 Raw Markdown", "💾 Export Data"])
        
        with tab_report:
            st.markdown(result_md)
            
        with tab_raw:
            st.text_area("Markdown Source", value=result_md, height=520)
            
        with tab_export:
            st.markdown("##### Download Generated Artifacts")
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                filename_md = f"video_{video_id}_{current_tpl}.md"
                st.download_button(
                    label="📥 Download Markdown (.md)",
                    data=result_md,
                    file_name=filename_md,
                    mime="text/markdown",
                    use_container_width=True
                )
                
            with col_d2:
                export_json = {
                    "video_url": display_url,
                    "video_id": video_id,
                    "template": current_tpl,
                    "model": st.session_state.get("latest_model", "gemini-3.7-flash"),
                    "elapsed_seconds": elapsed_sec,
                    "report_content": result_md
                }
                filename_json = f"video_{video_id}_{current_tpl}.json"
                st.download_button(
                    label="📥 Download JSON (.json)",
                    data=json.dumps(export_json, indent=2),
                    file_name=filename_json,
                    mime="application/json",
                    use_container_width=True
                )
    else:
        if not run_button:
            st.info("Click **'🚀 Run Analysis'** in the sidebar to process this video.")
