@echo off
color 1f
echo ■■■■■■■■■■■■■■■■■■■■■■■■
echo データ更新からherokuへのpushを行うバッチファイル
echo ■■■■■■■■■■■■■■■■■■■■■■■■
echo.
echo.
rem コンピュータ名の表示(hostname)
echo コンピュータ名
hostname
echo.
echo.
rem パスの指定
cd C:\Users\yhfhg\OneDrive\デスクトップ\Flask_app
echo データとモデルを更新します
python prep.py
echo データ・モデル更新完了
echo herokuへpushします
git add .
git commit -m "update"
git push heroku master
echo herokuへのpushが完了しました

call sub.bat >> result.log

pause