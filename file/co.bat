@baiecho off
setlocal enabledelayedexpansion

if not exist d:\123 md d:\123
pushd c:\123

for /f "tokens=*" %%i in ('dir/s/b') do (
if exist "d:\123\%%~nxi" (
    for /f %%j in ('dir/b "d:\123\%%~ni*%%~xi"^|find /c /v ".*"') do set /a n=%%j + 1
    copy "%%i" "d:\123\%%~ni!n!%%~xi"
) else copy "%%i" d:\123
)