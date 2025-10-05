@echo off
echo ===============================================
echo   コアクラス画像分類アプリ - 起動
echo ===============================================
echo.

echo 環境確認中...

REM 仮想環境存在確認
if not exist .venv (
    echo エラー: 仮想環境が見つかりません
    echo まず setup.bat を実行してセットアップを完了してください
    pause
    exit /b 1
)

REM モデルファイル確認
if not exist models (
    echo エラー: modelsフォルダが見つかりません
    echo 訓練済みモデルファイルを配置してください
    pause
    exit /b 1
)

echo 仮想環境を有効化中...
call .venv\Scripts\activate.bat

echo アプリケーション起動中...
python core_classifier_organizer.py

echo.
echo アプリケーションが終了しました
pause