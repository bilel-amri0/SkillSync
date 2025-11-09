"""
BERT-based Skills Extraction Model
Adapted from notebook for production use
"""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

# PyTorch imports with error handling
try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PyTorch not available: {e}")
    torch = None
    Dataset = None
    TORCH_AVAILABLE = False

# Transformers imports with error handling
try:
    from transformers import (
        BertTokenizerFast, 
        BertForTokenClassification,
        TrainingArguments, 
        Trainer,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Transformers not available: {e}")
    BertTokenizerFast = None
    BertForTokenClassification = None
    TrainingArguments = None
    Trainer = None
    pipeline = None
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

# NER Configuration
LABEL_NAMES = ["O", "B-SKILL", "I-SKILL"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_NAMES)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# Skills database
SKILLS_DATABASE = {
    "data_science": ["Python", "R", "SQL", "Machine Learning", "TensorFlow", "PyTorch", "Pandas", "NumPy", "Scikit-learn", "Jupyter"],
    "web_development": ["JavaScript", "React", "Node.js", "HTML", "CSS", "MongoDB", "Express", "Vue.js", "Angular", "TypeScript"],
    "mobile": ["Swift", "Kotlin", "React Native", "Flutter", "iOS", "Android", "Java", "Xamarin"],
    "devops": ["Docker", "Kubernetes", "Jenkins", "AWS", "Azure", "Terraform", "Linux", "Git", "CI/CD"],
    "backend": ["Java", "C#", "Go", "Rust", "Spring", "Django", "FastAPI", "MySQL", "PostgreSQL"],
    "frontend": ["React", "Vue.js", "Angular", "CSS", "Sass", "Webpack", "Redux", "Next.js"],
    "cloud": ["AWS", "Azure", "GCP", "Lambda", "EC2", "S3", "CloudFormation", "Serverless"]
}

class NERDataset:
    """Dataset class for BERT NER training"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 128):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for NERDataset")
        
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for NERDataset")
            
        item = self.data[idx]
        words = item['words'][:self.max_length-2]  # -2 for [CLS] and [SEP]
        labels = item['labels'][:self.max_length-2]

        # Tokenization with label alignment
        tokenized = self.tokenizer(
            words,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Align labels with tokens
        word_ids = tokenized.word_ids()
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)  # Special token
            elif word_idx != previous_word_idx:
                if word_idx < len(labels):
                    aligned_labels.append(labels[word_idx])
                else:
                    aligned_labels.append(LABEL2ID["O"])  # 'O' label
            else:
                aligned_labels.append(-100)  # Sub-token
            previous_word_idx = word_idx

        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }

class SkillsExtractorModel:
    """
    Production-ready BERT model for skills extraction
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", model_path: Optional[str] = None):
        self.device = None
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_name = model_name
        self.model_path = model_path
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.pipeline_extractor = None
        
        # Initialize if transformers is available
        if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
            self._initialize_model()
        else:
            logger.warning("Transformers/PyTorch not available, falling back to rule-based extraction")
            
    def _initialize_model(self):
        """Initialize BERT model and tokenizer"""
        try:
            if self.model_path and Path(self.model_path).exists():
                logger.info(f"Loading trained model from {self.model_path}")
                self.tokenizer = BertTokenizerFast.from_pretrained(self.model_path)
                self.model = BertForTokenClassification.from_pretrained(self.model_path)
            else:
                logger.info(f"Loading base model: {self.model_name}")
                self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)
                self.model = BertForTokenClassification.from_pretrained(
                    self.model_name,
                    num_labels=len(LABEL_NAMES),
                    label2id=LABEL2ID,
                    id2label=ID2LABEL
                )
            
            self.model.to(self.device)
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.tokenizer = None
            self.model = None
    
    def extract_skills_bert(self, text: str) -> List[str]:
        """
        Extract skills using BERT NER model
        """
        if not TORCH_AVAILABLE or self.model is None or self.tokenizer is None:
            return self.extract_skills_fallback(text)
            
        try:
            # Tokenize
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=2)

            # Extract tokens and labels
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            labels = [ID2LABEL[pred.item()] for pred in predictions[0]]

            # Extract skills
            skills = []
            current_skill = []

            for token, label in zip(tokens, labels):
                if token.startswith('##'):  # Sub-token
                    continue
                    
                if label == 'B-SKILL':
                    if current_skill:
                        skills.append(''.join(current_skill).replace('##', ''))
                    current_skill = [token]
                elif label == 'I-SKILL' and current_skill:
                    current_skill.append(token)
                else:
                    if current_skill:
                        skills.append(''.join(current_skill).replace('##', ''))
                        current_skill = []

            # Clean skills
            cleaned_skills = []
            for skill in skills:
                clean = skill.replace('##', '').strip()
                if len(clean) > 1 and clean not in ['[CLS]', '[SEP]', '[PAD]']:
                    cleaned_skills.append(clean)
                    
            return list(set(cleaned_skills))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error in BERT extraction: {e}")
            return self.extract_skills_fallback(text)
    
    def extract_skills_fallback(self, text: str) -> List[str]:
        """
        Fallback rule-based skills extraction
        """
        text_lower = text.lower()
        found_skills = []
        
        # Check all known skills
        for category, skills in SKILLS_DATABASE.items():
            for skill in skills:
                if skill.lower() in text_lower:
                    found_skills.append(skill)
        
        return list(set(found_skills))
    
    def extract_skills(self, text: str, use_bert: bool = True) -> Dict[str, any]:
        """
        Main skills extraction method with fallback
        """
        if use_bert and self.model is not None:
            bert_skills = self.extract_skills_bert(text)
            fallback_skills = self.extract_skills_fallback(text)
            
            # Combine results
            all_skills = list(set(bert_skills + fallback_skills))
            
            return {
                "skills": all_skills,
                "bert_skills": bert_skills,
                "fallback_skills": fallback_skills,
                "method": "bert_with_fallback",
                "confidence": "high" if len(bert_skills) > 0 else "medium"
            }
        else:
            fallback_skills = self.extract_skills_fallback(text)
            return {
                "skills": fallback_skills,
                "method": "rule_based",
                "confidence": "medium"
            }
    
    def categorize_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """
        Categorize extracted skills by domain
        """
        categorized = {category: [] for category in SKILLS_DATABASE.keys()}
        categorized["other"] = []
        
        for skill in skills:
            found_category = False
            for category, category_skills in SKILLS_DATABASE.items():
                if any(skill.lower() == cat_skill.lower() for cat_skill in category_skills):
                    categorized[category].append(skill)
                    found_category = True
                    break
            
            if not found_category:
                categorized["other"].append(skill)
        
        # Remove empty categories
        return {k: v for k, v in categorized.items() if v}
    
    def get_skill_suggestions(self, current_skills: List[str], target_level: str = "intermediate") -> List[Dict]:
        """
        Suggest related skills based on current skills
        """
        suggestions = []
        
        # Find domains of current skills
        current_domains = set()
        for skill in current_skills:
            for domain, domain_skills in SKILLS_DATABASE.items():
                if any(skill.lower() == ds.lower() for ds in domain_skills):
                    current_domains.add(domain)
        
        # Suggest skills from related domains
        for domain in current_domains:
            domain_skills = SKILLS_DATABASE[domain]
            missing_skills = [s for s in domain_skills if s not in current_skills]
            
            for skill in missing_skills[:3]:  # Top 3 suggestions per domain
                suggestions.append({
                    "skill": skill,
                    "domain": domain,
                    "reason": f"Complements your {domain} skills",
                    "priority": "high" if len(missing_skills) <= 5 else "medium"
                })
        
        return suggestions[:10]  # Limit to top 10
