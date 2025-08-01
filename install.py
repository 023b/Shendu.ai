#!/usr/bin/env python3
"""
Installation and setup verification script for Shendu Knowledge Vault
"""

def check_dependencies():
    """Check if all dependencies are installed"""
    required_packages = [
        ('sentence_transformers', 'sentence-transformers'),
        ('faiss', 'faiss-cpu'),
        ('nltk', 'nltk'), 
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('networkx', 'networkx'),
        ('matplotlib', 'matplotlib'),
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
        ('pydantic', 'pydantic'),
        ('frontmatter', 'python-frontmatter'),
        ('llama_cpp', 'llama-cpp-python')
    ]
    
    missing = []
    installed = []
    
    for package, pip_name in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
            installed.append(package)
        except ImportError:
            print(f"❌ {package} (install with: pip install {pip_name})")
            missing.append((package, pip_name))
            
    print(f"\n📊 Status: {len(installed)}/{len(required_packages)} packages installed")
    
    if missing:
        print(f"\n❌ Missing packages:")
        for package, pip_name in missing:
            print(f"   • {package} -> pip install {pip_name}")
        
        print(f"\n💡 Quick install command:")
        pip_names = [pip_name for _, pip_name in missing]
        print(f"pip install {' '.join(pip_names)}")
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True

def verify_model():
    """Verify that the LLM model is accessible"""
    try:
        from model import get_llama_model
        print("🤖 Loading LLM model...")
        llm = get_llama_model()
        print("✅ LLM model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ LLM model failed to load: {e}")
        print("💡 Make sure your model.py file is configured correctly")
        return False

def check_existing_files():
    """Check if existing Shendu files are present"""
    import os
    
    required_files = ['model.py', 'memory.py', 'logic.py']
    optional_files = ['api.py', 'chatbot.py']
    
    print("📁 Checking existing Shendu files...")
    
    missing_required = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_required.append(file)
    
    for file in optional_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"⚠️  {file} (optional)")
    
    if missing_required:
        print(f"\n❌ Missing required files: {', '.join(missing_required)}")
        return False
    
    return True

def main():
    print("🔧 Shendu Knowledge Vault Installation Check")
    print("=" * 50)
    
    print("\n📁 Checking existing Shendu files...")
    files_ok = check_existing_files()
    
    print("\n📦 Checking dependencies...")
    deps_ok = check_dependencies()
    
    if files_ok and deps_ok:
        print("\n🤖 Checking LLM model...")
        model_ok = verify_model()
        
        if model_ok:
            print("\n🎉 Installation verified! You're ready to go!")
            print("\n🚀 Next steps:")
            print("  1. python setup_vault.py     - Setup your knowledge vault")
            print("  2. python run_chatbot.py     - Start enhanced chatbot") 
            print("  3. python run_api.py         - Start API server")
        else:
            print("\n⚠️  Dependencies OK, but model needs attention")
    else:
        print("\n❌ Installation incomplete. Please fix the issues above.")

if __name__ == "__main__":
    main()