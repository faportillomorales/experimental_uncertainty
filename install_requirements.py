import subprocess
import sys
import platform

def check_python_version():
    """Verifica se a versão do Python é compatível (3.x)"""
    if sys.version_info[0] < 3:
        print("ERRO: Python 3.x é necessário para executar este programa.")
        print(f"Versão atual: {platform.python_version()}")
        sys.exit(1)
    print(f"✓ Python {platform.python_version()} detectado")

def install_package(package):
    """Instala um pacote Python usando pip"""
    try:
        print(f"\nInstalando {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} instalado com sucesso")
    except subprocess.CalledProcessError:
        print(f"✗ Erro ao instalar {package}")
        return False
    return True

def main():
    print("=== Instalação de Dependências para Análise de Dados Experimentais ===\n")
    
    # Verifica versão do Python
    check_python_version()
    
    # Lista de pacotes necessários
    packages = [
        "uncertainties",
        "pandas",
        "numpy",
        "matplotlib",
        "CoolProp",
        "openpyxl"
    ]
    
    # Instala cada pacote
    success = True
    for package in packages:
        if not install_package(package):
            success = False
    
    # Resumo final
    print("\n=== Resumo da Instalação ===")
    if success:
        print("\n✓ Todas as dependências foram instaladas com sucesso!")
        print("\nVocê pode agora executar o programa principal:")
        print("python exp_unc.py")
    else:
        print("\n✗ Algumas dependências não puderam ser instaladas.")
        print("Por favor, tente instalar manualmente os pacotes que falharam:")
        print("pip install uncertainties pandas numpy matplotlib CoolProp openpyxl")
    
    print("\nPressione Enter para sair...")
    input()

if __name__ == "__main__":
    main() 