"""
SkillSync ML Pipeline - NLG Experience Translator
==================================================
Real T5/FLAN-based text generation for experience rewriting.
NO templates - pure neural generation.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import torch

logger = logging.getLogger(__name__)

# Lazy imports
_transformers = None
_pipeline = None


def _get_transformers():
    """Lazy load transformers"""
    global _transformers
    if _transformers is None:
        try:
            import transformers
            _transformers = transformers
        except ImportError:
            logger.error("transformers not installed. Install: pip install transformers")
            raise
    return _transformers


@dataclass
class TranslationResult:
    """Result of experience translation"""
    original_text: str
    translated_text: str
    style: str
    model_used: str
    confidence: float
    timestamp: str


class NLGExperienceTranslator:
    """
    Real NLG-based experience translator using T5/FLAN models.
    
    Capabilities:
    - Rewrite experiences in different tones (professional, creative, technical)
    - Improve clarity and impact
    - Adapt language to job targets
    - Preserve semantic meaning while enhancing style
    """
    
    SUPPORTED_STYLES = {
        "professional": "Rewrite this experience in a professional, formal tone suitable for corporate roles:",
        "technical": "Rewrite this experience emphasizing technical details and methodologies:",
        "creative": "Rewrite this experience in a creative, engaging style:",
        "concise": "Rewrite this experience in a concise, bullet-point friendly format:",
        "impactful": "Rewrite this experience to emphasize achievements and measurable impact:",
        "executive": "Rewrite this experience in an executive-level tone focusing on leadership and strategy:"
    }
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: Optional[str] = None
    ):
        """
        Initialize NLG translator.
        
        Args:
            model_name: HuggingFace model name (flan-t5-base, flan-t5-large)
            device: 'cpu' or 'cuda'
        """
        self.model_name = model_name
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"🚀 Initializing NLGExperienceTranslator: {model_name}")
        logger.info(f"📱 Device: {self.device}")
        
        # Load model and tokenizer
        transformers = _get_transformers()
        
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_name}: {e}")
            raise
    
    def translate(
        self,
        text: str,
        style: str = "professional",
        max_length: int = 256,
        temperature: float = 0.7
    ) -> TranslationResult:
        """
        Translate/rewrite experience text using neural generation.
        
        Args:
            text: Original experience text
            style: Target style (professional, technical, creative, etc.)
            max_length: Maximum output length
            temperature: Sampling temperature (0.0-1.0)
            
        Returns:
            TranslationResult with rewritten text
        """
        if not text or not text.strip():
            logger.warning("Empty text provided")
            return TranslationResult(
                original_text="",
                translated_text="",
                style=style,
                model_used=self.model_name,
                confidence=0.0,
                timestamp=datetime.utcnow().isoformat()
            )
        
        # Get prompt for style
        if style not in self.SUPPORTED_STYLES:
            logger.warning(f"Unknown style '{style}', using 'professional'")
            style = "professional"
        
        prompt = self.SUPPORTED_STYLES[style]
        full_input = f"{prompt} {text}"
        
        logger.info(f"Translating text to '{style}' style ({len(text)} chars)")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                full_input,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    num_return_sequences=1
                )
            
            # Decode output
            translated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            ).strip()
            
            # Calculate confidence (heuristic based on length and quality)
            confidence = self._calculate_confidence(text, translated_text)
            
            logger.info(f"✅ Translation complete ({len(translated_text)} chars)")
            
            return TranslationResult(
                original_text=text,
                translated_text=translated_text,
                style=style,
                model_used=self.model_name,
                confidence=confidence,
                timestamp=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise
    
    def translate_batch(
        self,
        texts: List[str],
        style: str = "professional",
        max_length: int = 256,
        temperature: float = 0.7
    ) -> List[TranslationResult]:
        """
        Translate multiple texts (batch processing).
        
        Args:
            texts: List of texts to translate
            style: Target style
            max_length: Maximum output length
            temperature: Sampling temperature
            
        Returns:
            List of TranslationResult
        """
        results = []
        
        for text in texts:
            try:
                result = self.translate(text, style, max_length, temperature)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to translate text: {e}")
                # Add failed result
                results.append(TranslationResult(
                    original_text=text,
                    translated_text=text,  # Fallback to original
                    style=style,
                    model_used=self.model_name,
                    confidence=0.0,
                    timestamp=datetime.utcnow().isoformat()
                ))
        
        return results
    
    def translate_cv_experiences(
        self,
        cv_data: Dict[str, Any],
        target_style: str = "professional"
    ) -> Dict[str, Any]:
        """
        Translate all experiences in a CV.
        
        Args:
            cv_data: CV data with experience section
            target_style: Target style for translation
            
        Returns:
            CV data with translated experiences
        """
        translated_cv = cv_data.copy()
        
        # Translate summary if exists
        if cv_data.get("summary"):
            summary_result = self.translate(cv_data["summary"], target_style)
            translated_cv["summary"] = summary_result.translated_text
        
        # Translate experience descriptions
        if cv_data.get("experience"):
            exp = cv_data["experience"]
            translated_exp = []
            
            if isinstance(exp, list):
                for job in exp:
                    if isinstance(job, dict):
                        translated_job = job.copy()
                        
                        # Translate description
                        if job.get("description"):
                            desc_result = self.translate(job["description"], target_style)
                            translated_job["description"] = desc_result.translated_text
                        
                        translated_exp.append(translated_job)
                    else:
                        # String experience
                        result = self.translate(str(job), target_style)
                        translated_exp.append(result.translated_text)
                
                translated_cv["experience"] = translated_exp
            
            elif isinstance(exp, str):
                result = self.translate(exp, target_style)
                translated_cv["experience"] = result.translated_text
        
        logger.info("✅ CV experiences translated")
        return translated_cv
    
    def improve_bullet_points(
        self,
        bullet_points: List[str],
        emphasize: str = "impact"
    ) -> List[str]:
        """
        Improve resume bullet points to emphasize impact/achievements.
        
        Args:
            bullet_points: List of bullet points
            emphasize: What to emphasize (impact, technical, leadership)
            
        Returns:
            List of improved bullet points
        """
        improved = []
        
        for bullet in bullet_points:
            # Add action verb if missing
            prompt = f"Improve this resume bullet point to emphasize {emphasize} and start with a strong action verb: {bullet}"
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=256,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    temperature=0.7,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95
                )
            
            improved_bullet = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            improved.append(improved_bullet)
        
        return improved
    
    def adapt_to_job(
        self,
        experience_text: str,
        job_title: str,
        job_keywords: List[str]
    ) -> str:
        """
        Adapt experience text to match a specific job.
        
        Args:
            experience_text: Original experience
            job_title: Target job title
            job_keywords: Keywords from job description
            
        Returns:
            Adapted text
        """
        keywords_str = ", ".join(job_keywords[:5])
        
        prompt = (
            f"Rewrite this experience to match a {job_title} position, "
            f"emphasizing these keywords: {keywords_str}. "
            f"Experience: {experience_text}"
        )
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )
        
        adapted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        logger.info(f"✅ Experience adapted for {job_title}")
        return adapted_text
    
    def _calculate_confidence(self, original: str, translated: str) -> float:
        """
        Calculate confidence score for translation.
        
        Args:
            original: Original text
            translated: Translated text
            
        Returns:
            Confidence score (0-1)
        """
        # Heuristic confidence based on:
        # 1. Length similarity
        # 2. Non-empty output
        # 3. Different from input
        
        if not translated or len(translated) < 10:
            return 0.3
        
        if translated == original:
            return 0.5  # No change
        
        # Check length ratio
        len_ratio = len(translated) / max(len(original), 1)
        
        if 0.5 <= len_ratio <= 2.0:
            return 0.9  # Good length
        elif 0.3 <= len_ratio <= 3.0:
            return 0.7  # Acceptable
        else:
            return 0.5  # Too different
    
    def get_supported_styles(self) -> List[str]:
        """Get list of supported translation styles."""
        return list(self.SUPPORTED_STYLES.keys())


# Global singleton
_global_translator: Optional[NLGExperienceTranslator] = None


def get_translator(model_name: str = "google/flan-t5-base") -> NLGExperienceTranslator:
    """
    Get or create global translator (singleton).
    
    Args:
        model_name: Model to use
        
    Returns:
        NLGExperienceTranslator instance
    """
    global _global_translator
    
    if _global_translator is None:
        _global_translator = NLGExperienceTranslator(model_name=model_name)
    
    return _global_translator
