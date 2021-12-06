@echo off
setlocal enabledelayedexpansion
cd /d %~dp0

java -Xms1024m -Xmx1024m -cp "C:\Users\Motohiro\Documents\workspace\botFICE\FightingICE.jar;C:\Users\Motohiro\Documents\workspace\botFICE\lib\lwjgl\*;C:\Users\Motohiro\Documents\workspace\botFICE\lib\natives\windows\*;C:\Users\Motohiro\Documents\workspace\botFICE\lib\*;C:\Users\Motohiro\Documents\workspace\botFICE\data\ai\*;" Main -r 1 --limithp 400 400 --json