@echo off
color 1f
echo ������������������������������������������������
echo �f�[�^�X�V����heroku�ւ�push���s���o�b�`�t�@�C��
echo ������������������������������������������������
echo.
echo.
rem �R���s���[�^���̕\��(hostname)
echo �R���s���[�^��
hostname
echo.
echo.
rem �p�X�̎w��
cd C:\Users\yhfhg\OneDrive\�f�X�N�g�b�v\Flask_app
echo �f�[�^�ƃ��f�����X�V���܂�
python prep.py
echo �f�[�^�E���f���X�V����
echo heroku��push���܂�
git add .
git commit -m "update"
git push heroku master
echo heroku�ւ�push���������܂���

call sub.bat >> result.log

pause