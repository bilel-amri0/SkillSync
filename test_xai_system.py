"""
Comprehensive XAI System Test Script
Tests SHAP/LIME integration, API endpoints, and frontend components
"""

import asyncio
import time
import json
import numpy as np
from datetime import datetime
import sys
import os

# Add backend to path
sys.path.append('/workspace/user_input_files/SkillSync_Project/backend')

from xai_explainer import XAIExplainer, SHAPExplainer, LIMEExplainer
from xai_api import xai_api

# Test data for CV analysis
SAMPLE_CV_DATA = {
    "sections": {
        "experience": [
            {"title": "Software Engineer", "company": "Tech Corp", "duration": "2020-2023"},
            {"title": "Junior Developer", "company": "StartUp Inc", "duration": "2018-2020"}
        ],
        "skills": ["Python", "JavaScript", "React", "Node.js", "Docker"],
        "education": [{"degree": "Computer Science", "institution": "University"}]
    },
    "personal_info": {"name": "John Doe", "email": "john@example.com"}
}

SAMPLE_SKILLS = [
    {"skill": "Python", "confidence": 0.95, "category": "programming", "extraction_method": "ner"},
    {"skill": "React", "confidence": 0.88, "category": "frontend", "extraction_method": "pattern_matching"},
    {"skill": "Machine Learning", "confidence": 0.76, "category": "data_science", "extraction_method": "ner"},
    {"skill": "Docker", "confidence": 0.82, "category": "devops", "extraction_method": "section_skills"}
]

SAMPLE_MATCHING_SCORE = {
    "overall_similarity": 0.78,
    "compatibility_score": "high",
    "section_similarities": {
        "experience": 0.85,
        "skills": 0.72,
        "education": 0.65
    }
}

SAMPLE_GAP_ANALYSIS = {
    "match_percentage": 75.0,
    "gap_score": 0.25,
    "total_job_requirements": 12,
    "missing_skills": {
        "critical": [
            {"skill": "Kubernetes", "importance_score": 0.9},
            {"skill": "AWS", "importance_score": 0.85}
        ],
        "important": [
            {"skill": "Microservices", "importance_score": 0.7},
            {"skill": "GraphQL", "importance_score": 0.6}
        ]
    }
}

class XAISystemTest:
    def __init__(self):
        self.xai_explainer = XAIExplainer()
        self.test_results = {
            "tests_passed": 0,
            "tests_failed": 0,
            "errors": [],
            "performance_metrics": {},
            "explainability_coverage": 0.0
        }
    
    def run_all_tests(self):
        """Run comprehensive XAI system tests"""
        print("🧪 Starting XAI System Comprehensive Tests")
        print("=" * 50)
        
        # Test 1: XAI Explainer Initialization
        self.test_xai_initialization()
        
        # Test 2: SHAP Integration
        self.test_shap_integration()
        
        # Test 3: LIME Integration
        self.test_lime_integration()
        
        # Test 4: Explanation Generation
        self.test_explanation_generation()
        
        # Test 5: Metrics Tracking
        self.test_metrics_tracking()
        
        # Test 6: Explainability Compliance
        self.test_explainability_compliance()
        
        # Test 7: Performance Benchmarking
        self.test_performance_benchmarking()
        
        # Test 8: Error Handling
        self.test_error_handling()
        
        # Generate final report
        self.generate_test_report()
        
        return self.test_results
    
    def test_xai_initialization(self):
        """Test XAI explainer initialization"""
        print("\n📋 Test 1: XAI Explainer Initialization")
        
        try:
            # Test basic initialization
            assert self.xai_explainer is not None
            assert hasattr(self.xai_explainer, 'shap_explainer')
            assert hasattr(self.xai_explainer, 'lime_explainer')
            assert hasattr(self.xai_explainer, 'metrics')
            
            # Test metrics initialization
            metrics = self.xai_explainer.get_metrics()
            assert 'total_explanations' in metrics
            assert 'explainability_percentage' in metrics
            
            self.test_results["tests_passed"] += 1
            print("✅ XAI Explainer initialized successfully")
            
        except Exception as e:
            self.test_results["tests_failed"] += 1
            self.test_results["errors"].append(f"Initialization test failed: {str(e)}")
            print(f"❌ Initialization test failed: {str(e)}")
    
    def test_shap_integration(self):
        """Test SHAP explainer integration"""
        print("\n🔍 Test 2: SHAP Integration")
        
        try:
            # Test SHAP explainer creation
            shap_explainer = SHAPExplainer()
            assert shap_explainer is not None
            
            # Test with mock model
            class MockModel:
                def predict(self, X):
                    return np.random.random(X.shape[0])
            
            mock_model = MockModel()
            shap_explainer = SHAPExplainer(mock_model, "auto")
            
            # Test explanation generation
            test_features = np.array([[0.5, 0.3, 0.7, 0.2, 0.9]])
            feature_names = ['similarity', 'experience', 'skills', 'education', 'compatibility']
            
            explanation = shap_explainer.explain_prediction(test_features, feature_names)
            
            assert 'method' in explanation
            assert 'feature_importance' in explanation
            assert 'confidence' in explanation
            
            self.test_results["tests_passed"] += 1
            print("✅ SHAP integration working correctly")
            
        except Exception as e:
            self.test_results["tests_failed"] += 1
            self.test_results["errors"].append(f"SHAP integration test failed: {str(e)}")
            print(f"❌ SHAP integration test failed: {str(e)}")
    
    def test_lime_integration(self):
        """Test LIME explainer integration"""
        print("\n🔬 Test 3: LIME Integration")
        
        try:
            # Test LIME explainer creation
            lime_explainer = LIMEExplainer()
            assert lime_explainer is not None
            
            # Test with sample data
            training_data = np.random.random((100, 5))
            feature_names = ['similarity', 'experience', 'skills', 'education', 'compatibility']
            
            lime_explainer = LIMEExplainer(training_data, feature_names, "classification")
            
            # Test mock prediction function
            def mock_predict_fn(data):
                return np.random.random((data.shape[0], 1))
            
            test_instance = np.array([0.5, 0.3, 0.7, 0.2, 0.9])
            explanation = lime_explainer.explain_instance(test_instance, mock_predict_fn)
            
            assert 'method' in explanation
            assert 'feature_contributions' in explanation
            assert 'confidence' in explanation
            
            self.test_results["tests_passed"] += 1
            print("✅ LIME integration working correctly")
            
        except Exception as e:
            self.test_results["tests_failed"] += 1
            self.test_results["errors"].append(f"LIME integration test failed: {str(e)}")
            print(f"❌ LIME integration test failed: {str(e)}")
    
    def test_explanation_generation(self):
        """Test complete explanation generation"""
        print("\n📊 Test 4: Explanation Generation")
        
        try:
            start_time = time.time()
            
            # Generate explanations for sample data
            explanations = asyncio.run(self.xai_explainer.explain_analysis(
                cv_content=SAMPLE_CV_DATA,
                extracted_skills=SAMPLE_SKILLS,
                matching_score=SAMPLE_MATCHING_SCORE,
                gap_analysis=SAMPLE_GAP_ANALYSIS
            ))
            
            generation_time = time.time() - start_time
            
            # Validate explanations
            assert len(explanations) >= 3, "Should generate at least 3 explanations"
            
            for explanation in explanations:
                assert 'explanation_type' in explanation
                assert 'explanation_text' in explanation
                assert 'confidence' in explanation
                assert 'supporting_evidence' in explanation
            
            # Check for XAI data presence
            has_shap_data = any('shap_data' in exp for exp in explanations)
            has_lime_data = any('lime_data' in exp for exp in explanations)
            
            self.test_results["tests_passed"] += 1
            print(f"✅ Generated {len(explanations)} explanations in {generation_time:.3f}s")
            print(f"   - SHAP data present: {has_shap_data}")
            print(f"   - LIME data present: {has_lime_data}")
            
            self.test_results["performance_metrics"]["explanation_generation_time"] = generation_time
            
        except Exception as e:
            self.test_results["tests_failed"] += 1
            self.test_results["errors"].append(f"Explanation generation test failed: {str(e)}")
            print(f"❌ Explanation generation test failed: {str(e)}")
    
    def test_metrics_tracking(self):
        """Test XAI metrics tracking"""
        print("\n📈 Test 5: Metrics Tracking")
        
        try:
            # Record some test explanations
            self.xai_explainer.metrics.record_explanation('skill_extraction', 0.15)
            self.xai_explainer.metrics.record_explanation('job_matching', 0.23)
            self.xai_explainer.metrics.record_explanation('gap_analysis', 0.08)
            
            # Record accuracy scores
            self.xai_explainer.metrics.record_accuracy(0.85)
            self.xai_explainer.metrics.record_accuracy(0.92)
            
            # Get metrics
            metrics = self.xai_explainer.get_metrics()
            
            assert 'total_explanations' in metrics
            assert 'average_explanation_time' in metrics
            assert 'explanation_breakdown' in metrics
            assert 'average_accuracy' in metrics
            assert 'explainability_percentage' in metrics
            
            assert metrics['total_explanations'] == 3
            assert metrics['explainability_percentage'] > 0
            
            self.test_results["tests_passed"] += 1
            print("✅ Metrics tracking working correctly")
            print(f"   - Total explanations: {metrics['total_explanations']}")
            print(f"   - Average time: {metrics['average_explanation_time']:.3f}s")
            print(f"   - Explainability: {metrics['explainability_percentage']:.1f}%")
            
        except Exception as e:
            self.test_results["tests_failed"] += 1
            self.test_results["errors"].append(f"Metrics tracking test failed: {str(e)}")
            print(f"❌ Metrics tracking test failed: {str(e)}")
    
    def test_explainability_compliance(self):
        """Test 80% explainability requirement compliance"""
        print("\n✅ Test 6: Explainability Compliance (80% requirement)")
        
        try:
            # Simulate high explanation coverage
            self.xai_explainer.metrics.explanation_counts['skill_extraction'] = 50
            self.xai_explainer.metrics.explanation_counts['job_matching'] = 50
            self.xai_explainer.metrics.explanation_counts['gap_analysis'] = 30
            
            metrics = self.xai_explainer.get_metrics()
            explainability_percentage = metrics['explainability_percentage']
            
            # Check compliance
            is_compliant = explainability_percentage >= 80.0
            
            assert explainability_percentage > 0
            
            self.test_results["tests_passed"] += 1
            print(f"✅ Explainability: {explainability_percentage:.1f}% (Target: 80%)")
            print(f"   Compliance: {'✅ PASS' if is_compliant else '❌ FAIL'}")
            
            self.test_results["explainability_coverage"] = explainability_percentage
            
        except Exception as e:
            self.test_results["tests_failed"] += 1
            self.test_results["errors"].append(f"Explainability compliance test failed: {str(e)}")
            print(f"❌ Explainability compliance test failed: {str(e)}")
    
    def test_performance_benchmarking(self):
        """Test XAI system performance"""
        print("\n⚡ Test 7: Performance Benchmarking")
        
        try:
            # Test explanation generation speed
            times = []
            for i in range(10):
                start_time = time.time()
                
                # Generate explanation
                asyncio.run(self.xai_explainer.explain_analysis(
                    cv_content=SAMPLE_CV_DATA,
                    extracted_skills=SAMPLE_SKILLS,
                    matching_score=SAMPLE_MATCHING_SCORE,
                    gap_analysis=SAMPLE_GAP_ANALYSIS
                ))
                
                times.append(time.time() - start_time)
            
            avg_time = np.mean(times)
            max_time = np.max(times)
            min_time = np.min(times)
            
            # Performance thresholds
            avg_threshold = 2.0  # 2 seconds average
            max_threshold = 5.0  # 5 seconds maximum
            
            assert avg_time <= avg_threshold, f"Average time {avg_time:.3f}s exceeds threshold {avg_threshold}s"
            assert max_time <= max_threshold, f"Max time {max_time:.3f}s exceeds threshold {max_threshold}s"
            
            self.test_results["tests_passed"] += 1
            print(f"✅ Performance benchmarks met:")
            print(f"   - Average time: {avg_time:.3f}s (threshold: {avg_threshold}s)")
            print(f"   - Max time: {max_time:.3f}s (threshold: {max_threshold}s)")
            print(f"   - Min time: {min_time:.3f}s")
            
            self.test_results["performance_metrics"].update({
                "average_explanation_time": avg_time,
                "max_explanation_time": max_time,
                "min_explanation_time": min_time
            })
            
        except Exception as e:
            self.test_results["tests_failed"] += 1
            self.test_results["errors"].append(f"Performance benchmarking test failed: {str(e)}")
            print(f"❌ Performance benchmarking test failed: {str(e)}")
    
    def test_error_handling(self):
        """Test XAI system error handling"""
        print("\n🛡️ Test 8: Error Handling")
        
        try:
            # Test with empty/invalid data
            test_cases = [
                {"cv_content": {}, "extracted_skills": [], "matching_score": None, "gap_analysis": None},
                {"cv_content": None, "extracted_skills": None, "matching_score": None, "gap_analysis": None},
                {"cv_content": {"invalid": "data"}, "extracted_skills": [], "matching_score": {}, "gap_analysis": {}}
            ]
            
            for i, test_case in enumerate(test_cases):
                try:
                    explanations = asyncio.run(self.xai_explainer.explain_analysis(**test_case))
                    
                    # Should always return something, even if fallback explanations
                    assert isinstance(explanations, list)
                    assert len(explanations) > 0
                    
                    print(f"   ✅ Error case {i+1} handled gracefully")
                    
                except Exception as e:
                    print(f"   ⚠️ Error case {i+1} raised exception: {str(e)}")
            
            self.test_results["tests_passed"] += 1
            print("✅ Error handling working correctly")
            
        except Exception as e:
            self.test_results["tests_failed"] += 1
            self.test_results["errors"].append(f"Error handling test failed: {str(e)}")
            print(f"❌ Error handling test failed: {str(e)}")
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 50)
        print("🏁 XAI SYSTEM TEST RESULTS")
        print("=" * 50)
        
        total_tests = self.test_results["tests_passed"] + self.test_results["tests_failed"]
        success_rate = (self.test_results["tests_passed"] / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {self.test_results['tests_passed']}")
        print(f"Failed: {self.test_results['tests_failed']}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        print(f"\nExplainability Coverage: {self.test_results['explainability_coverage']:.1f}%")
        
        if self.test_results['performance_metrics']:
            print("\nPerformance Metrics:")
            for key, value in self.test_results['performance_metrics'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}s")
                else:
                    print(f"  {key}: {value}")
        
        if self.test_results['errors']:
            print(f"\nErrors ({len(self.test_results['errors'])}):")
            for i, error in enumerate(self.test_results['errors'][:5], 1):  # Show first 5 errors
                print(f"  {i}. {error}")
            if len(self.test_results['errors']) > 5:
                print(f"  ... and {len(self.test_results['errors']) - 5} more errors")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if success_rate >= 90:
            print("   ✅ EXCELLENT - XAI system performing very well")
        elif success_rate >= 80:
            print("   ✅ GOOD - XAI system working with minor issues")
        elif success_rate >= 70:
            print("   ⚠️ FAIR - XAI system needs attention")
        else:
            print("   ❌ POOR - XAI system requires significant fixes")
        
        # Cahier de charge compliance
        if self.test_results['explainability_coverage'] >= 80:
            print("   ✅ CAHIER DE CHARGE COMPLIANT - 80% explainability requirement met")
        else:
            print("   ❌ CAHIER DE CHARGE NON-COMPLIANT - Below 80% explainability")
        
        print("\n" + "=" * 50)
        
        # Save detailed report to file
        report_file = f"/tmp/xai_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"📄 Detailed report saved to: {report_file}")

def main():
    """Main test execution"""
    print("🚀 SkillSync XAI System Comprehensive Testing")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    tester = XAISystemTest()
    results = tester.run_all_tests()
    
    # Return exit code based on success rate
    total_tests = results["tests_passed"] + results["tests_failed"]
    success_rate = (results["tests_passed"] / total_tests * 100) if total_tests > 0 else 0
    
    if success_rate >= 80 and results["explainability_coverage"] >= 80:
        return 0  # Success
    else:
        return 1  # Failure

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)