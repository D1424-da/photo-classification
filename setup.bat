@echo off
echo ===============================================
echo   コアクラス画像分類アプリ - セットアップ
echo ===============================================
echo.

echo [1/4] Python環境確認中...
python --version
if %errorlevel% neq 0 (
    echo エラー: Pythonがインストールされていません
    echo Python 3.8以上をインストールしてください
    pause
    exit /b 1
)

echo [2/4] 仮想環境作成中...
if exist .venv (
    echo 既存の仮想環境を削除中...
    rmdir /s /q .venv
)
python -m venv .venv
if %errorlevel% neq 0 (
    echo エラー: 仮想環境の作成に失敗しました
    pause
    exit /b 1
)

echo [3/4] 仮想環境有効化中...
call .venv\Scripts\activate.bat

echo [4/4] 依存関係インストール中...
pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo エラー: 依存関係のインストールに失敗しました
    pause
    exit /b 1
)

echo.
echo ===============================================
echo   セットアップ完了！
echo ===============================================
echo.
echo 次の手順:
echo 1. start_app.bat をダブルクリックしてアプリを起動
echo 2. または: python core_classifier_organizer.py
echo.
pause