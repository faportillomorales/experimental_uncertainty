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
    print("Cobre: exp_unc.py, plot_tool.py (e leitura Excel .xlsx / .xlsm / .xls).\n")

    # Verifica versão do Python
    check_python_version()

    # Pacotes de terceiros (stdlib: os, pathlib, warnings, contextlib — não instalar)
    # - uncertainties: exp_unc.py
    # - pandas, numpy, matplotlib: exp_unc.py e plot_tool.py
    # - CoolProp: plot_tool.py (Re_sg, propriedades do fluido) e uso em exp_unc se aplicável
    # - openpyxl: Excel .xlsx / .xlsm (pandas.read_excel)
    # - xlrd<2: Excel binário .xls (plot_tool aceita extensão .xls; xlrd 2+ não lê .xls)
    packages = [
        "uncertainties",
        "pandas",
        "numpy",
        "matplotlib",
        "CoolProp",
        "openpyxl",
        "xlrd<2",
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
        print("\nPode executar, por exemplo:")
        print("  python exp_unc.py")
        print("  python plot_tool.py")
    else:
        print("\n✗ Algumas dependências não puderam ser instaladas.")
        print("Instale manualmente:")
        print(
            'pip install uncertainties pandas numpy matplotlib CoolProp openpyxl "xlrd<2"'
        )
    
    print("\nPressione Enter para sair...")
    input()

if __name__ == "__main__":
    main() 