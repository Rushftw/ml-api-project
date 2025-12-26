import requests
import time
import json

# Models to compare
MODELS = ["gemma:2b", "mistral", "llama2:7b", "phi"]  # Add phi if you re-downloaded it

# Test questions
TEST_QUESTIONS = [
    {
        "category": "Factual Knowledge",
        "question": "What is the capital of France?",
        "expected": "Paris"
    },
    {
        "category": "Logic/Math",
        "question": "What is 15 + 27?",
        "expected": "42"
    },
    {
        "category": "Common Sense",
        "question": "Can a bird fly?",
        "expected": "Yes"
    },
    {
        "category": "Hallucination Test",
        "question": "What is the population of the fictional city of Gotham?",
        "expected": "I don't know"  # Should admit not knowing
    },
    {
        "category": "Instruction Following",
        "question": "Repeat the word 'hello' three times.",
        "expected": "hello hello hello"
    }
]

def test_model(model_name, question, expected):
    """Test a single model with one question"""
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:8001/generate",
            json={
                "prompt": question,
                "model": model_name,
                "max_tokens": 50
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("generated_text", "").strip().lower()
            duration = result.get("total_duration", 0) / 1_000_000_000  # Convert to seconds
            
            # Simple accuracy check
            accuracy = 1.0 if expected.lower() in generated_text else 0.0
            
            # Hallucination check (if asks about unknown, should admit)
            if "fictional" in question.lower() or "gotham" in question.lower():
                hallucination = 0.0 if "don't know" in generated_text or "fictional" in generated_text else 1.0
            else:
                hallucination = 0.0
            
            return {
                "success": True,
                "response": generated_text[:100],  # First 100 chars
                "duration": duration,
                "accuracy": accuracy,
                "hallucination": hallucination
            }
        else:
            return {
                "success": False,
                "error": f"API error: {response.status_code}"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        total_time = time.time() - start_time

def run_comparison():
    """Run comparison tests for all models"""
    results = {}
    
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Testing model: {model}")
        print('='*60)
        
        model_results = {
            "total_accuracy": 0,
            "total_hallucination": 0,
            "total_duration": 0,
            "passed_tests": 0,
            "details": []
        }
        
        for i, test in enumerate(TEST_QUESTIONS):
            print(f"\nTest {i+1}: {test['category']}")
            print(f"Question: {test['question']}")
            
            result = test_model(model, test['question'], test['expected'])
            
            if result["success"]:
                model_results["total_accuracy"] += result["accuracy"]
                model_results["total_hallucination"] += result["hallucination"]
                model_results["total_duration"] += result["duration"]
                model_results["passed_tests"] += 1
                
                print(f"Response: {result['response']}")
                print(f"Duration: {result['duration']:.2f}s")
                print(f"Accuracy: {'✓' if result['accuracy'] > 0 else '✗'}")
                print(f"Hallucination: {'✗' if result['hallucination'] > 0 else '✓'}")
                
                model_results["details"].append({
                    "question": test['question'],
                    "response": result['response'],
                    "duration": result['duration'],
                    "accuracy": result['accuracy'],
                    "hallucination": result['hallucination']
                })
            else:
                print(f"Failed: {result['error']}")
                model_results["details"].append({
                    "question": test['question'],
                    "error": result['error']
                })
        
        # Calculate averages
        if model_results["passed_tests"] > 0:
            model_results["avg_accuracy"] = model_results["total_accuracy"] / model_results["passed_tests"]
            model_results["avg_hallucination"] = model_results["total_hallucination"] / model_results["passed_tests"]
            model_results["avg_duration"] = model_results["total_duration"] / model_results["passed_tests"]
        else:
            model_results["avg_accuracy"] = 0
            model_results["avg_hallucination"] = 0
            model_results["avg_duration"] = 0
        
        results[model] = model_results
        
        print(f"\nSummary for {model}:")
        print(f"  Average Accuracy: {model_results['avg_accuracy']:.2%}")
        print(f"  Average Hallucination: {model_results['avg_hallucination']:.2%}")
        print(f"  Average Duration: {model_results['avg_duration']:.2f}s")
    
    return results

def choose_best_model(results):
    """Select best model based on weighted criteria"""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print('='*60)
    
    scores = {}
    
    for model, data in results.items():
        if data["passed_tests"] > 0:
            # Weighted score: 40% accuracy, 40% anti-hallucination, 20% speed
            accuracy_score = data["avg_accuracy"] * 0.4
            hallucination_score = (1 - data["avg_hallucination"]) * 0.4
            
            # Speed score: faster is better (normalized)
            max_duration = max([r["avg_duration"] for r in results.values() if r["passed_tests"] > 0])
            speed_score = (1 - (data["avg_duration"] / max_duration)) * 0.2 if max_duration > 0 else 0
            
            total_score = accuracy_score + hallucination_score + speed_score
            scores[model] = total_score
            
            print(f"\n{model}:")
            print(f"  Accuracy: {data['avg_accuracy']:.2%}")
            print(f"  Hallucination: {data['avg_hallucination']:.2%}")
            print(f"  Speed: {data['avg_duration']:.2f}s")
            print(f"  Overall Score: {total_score:.3f}")
    
    if scores:
        best_model = max(scores, key=scores.get)
        print(f"\n{'='*60}")
        print(f"RECOMMENDED FOR AI AGENT: {best_model}")
        print(f"Score: {scores[best_model]:.3f}")
        print('='*60)
        
        justification = f"""
        Justification for choosing {best_model}:
        1. Highest overall score ({scores[best_model]:.3f}) balancing accuracy, low hallucination, and speed
        2. Accuracy: {results[best_model]['avg_accuracy']:.2%} on test questions
        3. Hallucination rate: {results[best_model]['avg_hallucination']:.2%} (lower is better)
        4. Response time: {results[best_model]['avg_duration']:.2f} seconds
        5. Best suited for agent tasks requiring reliability and speed
        """
        print(justification)
        
        # Save results to file
        with open("model_comparison_results.json", "w") as f:
            json.dump({
                "results": results,
                "scores": scores,
                "best_model": best_model,
                "justification": justification
            }, f, indent=2)
        
        print("\nResults saved to 'model_comparison_results.json'")
        
        return best_model, justification
    else:
        print("No models successfully completed tests.")
        return None, None

if __name__ == "__main__":
    print("Starting LLM Comparison Tests...")
    print("Make sure LLM API is running on http://localhost:8001")
    print("Models to test:", MODELS)
    
    # Check API
    try:
        health = requests.get("http://localhost:8001/health", timeout=5)
        if health.status_code == 200:
            print("\nAPI is healthy. Starting tests...")
            results = run_comparison()
            best_model, justification = choose_best_model(results)
        else:
            print("API is not healthy. Start it with: python llm_api.py")
    except:
        print("Cannot connect to API. Start it with: python llm_api.py")