import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import math

# Try importing features. If it fails, help instructions will be shown
try:
    from features import (
        extractFeatures,
        hasDictionaryPattern,
        hasKeyboardPattern,
        hasSequencePattern
    )
    import features as feat_module
    FEATURES_MODULE_AVAILABLE = True
except Exception as e:
    FEATURES_MODULE_AVAILABLE = False
    FEATURES_IMPORT_ERROR = str(e)

# Streamlit Page Config
st.set_page_config(
    page_title="HyPSA | Hybrid Password Strength Analyzer",
    page_icon="🔒",
    layout="centered"
)

# Styled header
st.markdown("""
    <div style='text-align: center; padding-bottom: 20px;'>
        <h1 style='color: #1E3A8A; margin-bottom: 5px;'>🔒 HyPSA</h1>
        <h3 style='color: #4B5563; font-weight: normal; margin-top: 0;'>Hybrid Password Strength Analyzer</h3>
        <p style='color: #6B7280; font-size: 14px;'>Combining Shannon Entropy, Ensemble Machine Learning, and Rule-Based Fallback Penalties</p>
    </div>
""", unsafe_allow_html=True)

# Cache Model Loading
@st.cache_resource
def load_ml_models():
    models_path = "models"
    try:
        lr = joblib.load(os.path.join(models_path, "lr_model.pkl"))
        rf = joblib.load(os.path.join(models_path, "rf_model.pkl"))
        xgb = joblib.load(os.path.join(models_path, "xgb_model.pkl"))
        scaler = joblib.load(os.path.join(models_path, "scaler.pkl"))
        return lr, rf, xgb, scaler, True, ""
    except Exception as e:
        return None, None, None, None, False, str(e)

# Check dependencies first
if not FEATURES_MODULE_AVAILABLE:
    st.error("❌ **Dependency Error: Failed to import features.py**")
    st.info(f"**Error details**: {FEATURES_IMPORT_ERROR}")
    st.markdown("""
        ### How to fix:
        1. Make sure you are in the project folder: `c:\\Users\\HP\\OneDrive\\Documents\\8TH SEM\\FYRP\\HyPSA-main`
        2. Verify that **RapidFuzz** is installed: `pip install rapidfuzz`
        3. Make sure the `datasets` folder exists with `10000_common_passwords.csv`.
    """)
else:
    lr, rf, xgb, scaler, models_loaded, model_error = load_ml_models()

    if not models_loaded:
        st.warning("⚠️ **Trained Machine Learning Models Not Found or Mismatched**")
        st.markdown(f"""
            The pre-trained models could not be loaded from the `models/` folder.
            *Error details*: `{model_error}`
            
            ### How to retrain models on this computer:
            You can retrain the models in under 60 seconds by running these scripts in your terminal:
            ```bash
            python freezeDataset.py
            python model.py
            ```
            Once completed, reload this page!
        """)
        
        # Simple Fallback mode using only Shannon Entropy & Rules
        st.info("💡 Running in **Fallback Mode** (Statistical Randomness + Rules only)")
        
        password = st.text_input("Enter Password to Evaluate", type="password", key="pwd_fallback")
        
        if password:
            features = extractFeatures(password)
            entropy = features["shannonEntropy"]
            length = len(password)
            
            kb = hasKeyboardPattern(password)
            seq = hasSequencePattern(password)
            dict_flag = hasDictionaryPattern(password)
            
            # Simple scoring
            base_score = min(1.0, (entropy * 0.6) + (min(1.0, length/14) * 0.4))
            penalty = 1.0
            reasons = []
            
            if dict_flag:
                penalty *= 0.7
                reasons.append("Common dictionary word pattern")
            if kb:
                penalty *= 0.9
                reasons.append("Adjacent keyboard sequence (e.g. qwerty)")
            if seq:
                penalty *= 0.9
                reasons.append("Sequential character run (e.g. 123, abc)")
                
            final_score = base_score * penalty
            
            if final_score >= 0.55:
                verdict = "STRONG"
                color = "green"
            elif final_score >= 0.35:
                verdict = "MEDIUM"
                color = "orange"
            else:
                verdict = "WEAK"
                color = "red"
                
            st.markdown(f"### Verdict: <span style='color:{color}; font-weight:bold;'>{verdict}</span>", unsafe_allow_html=True)
            st.metric("Shannon Entropy Score", f"{entropy:.2f}")
            st.write(f"**Password Length**: {length} characters")
            if reasons:
                st.write("**Identified Weaknesses:**")
                for r in reasons:
                    st.write(f"- ⚠️ {r}")

    else:
        # Full Hybrid Prediction Mode
        password = st.text_input("Enter Password to Evaluate", type="password", help="Type your password here to inspect its strength.")

        if password:
            # Feature extraction
            features = extractFeatures(password)
            FEATURE_ORDER = ["length", "maxRepetition", "shannonEntropy", "sequenceCount", "dictionaryScore", "keyboardScore"]
            X_df = pd.DataFrame([[features[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
            
            # Model inference
            X_scaled = scaler.transform(X_df)
            
            lr_prob = lr.predict_proba(X_scaled)[0]
            rf_prob = rf.predict_proba(X_df)[0]
            xgb_prob = xgb.predict_proba(X_df)[0]
            
            # Compute individual model scores (weighted sums of probabilities)
            # label classes: 0 = Weak, 1 = Medium, 2 = Strong
            lr_score = 0.0 * lr_prob[0] + 0.5 * lr_prob[1] + 1.0 * lr_prob[2]
            rf_score = 0.0 * rf_prob[0] + 0.5 * rf_prob[1] + 1.0 * rf_prob[2]
            xgb_score = 0.0 * xgb_prob[0] + 0.5 * xgb_prob[1] + 1.0 * xgb_prob[2]
            
            # Ensemble combination (weights matching predict.py)
            ml_score = 0.2 * lr_score + 0.5 * rf_score + 0.3 * xgb_score
            entropy_score = features["shannonEntropy"]
            
            # Base hybrid score combining ML predictions and Shannon Entropy
            final_score = (0.4 * entropy_score) + (0.6 * ml_score)
            
            # Rule-based heuristics
            dict_flag = hasDictionaryPattern(password)
            kb_flag = hasKeyboardPattern(password)
            seq_flag = hasSequencePattern(password)
            
            penalty = 1.0
            reasons = []
            is_highly_predictable = False
            
            # Dictionary + structural patterns penalty
            if dict_flag and (features["sequenceCount"] > 0.4 or features["keyboardScore"] > 0.4):
                is_highly_predictable = True
                final_score = 0.15
                reasons.append("Highly predictable combination (dictionary word + structural pattern)")
            
            if not is_highly_predictable:
                if dict_flag and len(password) < 12:
                    penalty *= 0.75
                    reasons.append("Dictionary word/pattern detected in shorter password")
                if features["sequenceCount"] > 0.4:
                    penalty *= 0.9
                    reasons.append("Sequential character runs detected (e.g., abc, 123)")
                if features["keyboardScore"] > 0.4:
                    penalty *= 0.9
                    reasons.append("Physical keyboard adjacency patterns detected (e.g., qwerty, asdf)")
                
                final_score *= penalty
            
            # Bounds correction
            final_score = min(1.0, max(0.0, final_score))
            
            # Verdict boundary
            if final_score >= 0.50:
                verdict = "STRONG"
                color = "#10B981"  # Emerald Green
                bg_color = "#D1FAE5"
                text_color = "#065F46"
            elif final_score >= 0.30:
                verdict = "MEDIUM"
                color = "#F59E0B"  # Amber/Orange
                bg_color = "#FEF3C7"
                text_color = "#92400E"
            else:
                verdict = "WEAK"
                color = "#EF4444"  # Red
                bg_color = "#FEE2E2"
                text_color = "#991B1B"
            
            # Render verdict banner
            st.markdown(f"""
                <div style='background-color: {bg_color}; padding: 15px; border-radius: 8px; border-left: 5px solid {color}; margin-top: 15px; margin-bottom: 20px; text-align: center;'>
                    <span style='color: {text_color}; font-size: 20px; font-weight: bold;'>Verdict: {verdict}</span>
                    <br/>
                    <span style='color: {text_color}; font-size: 14px;'>Hybrid Security Score: <b>{final_score * 100:.1f} / 100</b></span>
                </div>
            """, unsafe_allow_html=True)
            
            # Metrics columns
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Shannon Entropy (Randomness)", f"{entropy_score:.2f}")
            with m_col2:
                st.metric("Ensemble ML Score", f"{ml_score:.2f}")
            with m_col3:
                st.metric("Password Length", f"{len(password)}")
                
            # Weakness Alert Box
            if reasons:
                st.markdown("### ⚠️ Security Warnings & Rule Violations")
                for reason in reasons:
                    st.markdown(f"- **{reason}**")
            else:
                st.markdown("### ✅ No Rule Violations Detected")
                st.markdown("Your password successfully avoided common dictionary words, sequential sweeps, and keyboard adjacencies.")
                
            # Details Tabs
            st.markdown("---")
            tab1, tab2, tab3 = st.tabs(["📊 Feature breakdown", "🤖 Machine Learning details", "💡 Recommendations"])
            
            with tab1:
                st.markdown("#### Extracted Quantitative Features (Normalized [0-1])")
                
                # Raw feature table
                feat_data = {
                    "Feature Name": ["Password Length", "Max Repetition Rate", "Normalized Shannon Entropy", "Sequence Count", "Dictionary Match Score", "Keyboard Pattern Score"],
                    "Value": [
                        f"{features['length']:.2f} ({len(password)} chars)",
                        f"{features['maxRepetition']:.2f}",
                        f"{features['shannonEntropy']:.2f}",
                        f"{features['sequenceCount']:.2f}",
                        f"{features['dictionaryScore']:.2f}",
                        f"{features['keyboardScore']:.2f}"
                    ],
                    "Security Impact": [
                        "Primary protection against Brute Force.",
                        "Higher repetition decreases randomness.",
                        "Direct mathematical measurement of entropy.",
                        "Tracks predictable runs like 123, abc, or aceg.",
                        "Measures similarity to common credentials.",
                        "Measures spatial keyboard swipe predictability."
                    ]
                }
                st.table(pd.DataFrame(feat_data))
                
            with tab2:
                st.markdown("#### ML Ensemble Class Probability Distributions")
                
                # Show model predictions
                pred_df = pd.DataFrame({
                    "Logistic Regression": lr_prob,
                    "Random Forest": rf_prob,
                    "XGBoost": xgb_prob
                }, index=["Weak (0)", "Medium (1)", "Strong (2)"])
                
                st.dataframe(pred_df.style.format("{:.4f}").background_gradient(cmap="Blues"))
                st.write("**Model Prediction Scores (Weighted Value):**")
                st.write(f"- **Logistic Regression Score**: `{lr_score:.3f}` (Weight: 20%)")
                st.write(f"- **Random Forest Score**: `{rf_score:.3f}` (Weight: 50%)")
                st.write(f"- **XGBoost Score**: `{xgb_score:.3f}` (Weight: 30%)")
                
            with tab3:
                st.markdown("#### Actionable Steps to Improve Strength")
                
                suggestions = []
                if len(password) < 12:
                    suggestions.append("👉 **Make it longer**: Increase the length to at least 12–16 characters. Length is the single most effective barrier against modern GPU-based cracking.")
                if dict_flag:
                    suggestions.append("👉 **Avoid dictionary words**: Do not use common dictionary words or names, even with number substitutions like 'a' to '@'. Attackers precompute these variations.")
                if features["sequenceCount"] > 0.3:
                    suggestions.append("👉 **Remove character sequences**: Avoid runs like '123', 'abc', or 'qwe'. Introduce non-sequential characters instead.")
                if features["keyboardScore"] > 0.3:
                    suggestions.append("👉 **Break spatial keyboard shapes**: Avoid drawing lines or patterns on your keyboard (like 'qwerty' or 'asdfgh'). Combine keys from disparate regions.")
                if features["maxRepetition"] > 0.3:
                    suggestions.append("👉 **Limit repeated characters**: Avoid multiple identical or repeating symbols (e.g., 'aaaa' or '1111').")
                    
                if not suggestions:
                    st.success("🌟 Your password is exceptionally robust! No recommendations needed.")
                else:
                    for sugg in suggestions:
                        st.markdown(sugg)
                        
# Footer
st.markdown("""
    <hr style='margin-top: 40px;'/>
    <div style='text-align: center; color: #9CA3AF; font-size: 12px; padding-bottom: 20px;'>
        HyPSA Framework &copy; 2026 | Supervised by Dr. Suneeta Satpathy
    </div>
""", unsafe_allow_html=True)
