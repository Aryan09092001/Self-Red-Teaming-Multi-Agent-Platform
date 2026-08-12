@echo off
setlocal

set REGION=us-east-1
rem S3 bucket names are globally unique across all AWS accounts, so the account id
rem is appended to keep this one ours. Must match the backend block in terraform\main.tf.
for /f "delims=" %%i in ('aws sts get-caller-identity --query Account --output text') do set ACCOUNT_ID=%%i
set BUCKET=research-agent-tfstate-%ACCOUNT_ID%
set TABLE=research-agent-tf-locks

echo Creating S3 bucket: %BUCKET% in region: %REGION%

rem Only "we already own it" is safe to ignore. Any other failure (name taken by
rem another account, bad credentials) must stop the script, not be swallowed.
aws s3api create-bucket --bucket %BUCKET% --region %REGION% 2>"%TEMP%\tfboot_err.txt"
if %errorlevel% equ 0 (
    echo Bucket created.
) else (
    findstr /C:"BucketAlreadyOwnedByYou" "%TEMP%\tfboot_err.txt" >nul
    if errorlevel 1 (
        echo ERROR: could not create bucket %BUCKET%
        type "%TEMP%\tfboot_err.txt"
        exit /b 1
    )
    echo Bucket already exists and is owned by this account, continuing.
)

echo Enabling versioning...
aws s3api put-bucket-versioning --bucket %BUCKET% --versioning-configuration Status=Enabled

echo Blocking public access...
aws s3api put-public-access-block --bucket %BUCKET% --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

echo Enabling server-side encryption...
aws s3api put-bucket-encryption --bucket %BUCKET% --server-side-encryption-configuration "{\"Rules\":[{\"ApplyServerSideEncryptionByDefault\":{\"SSEAlgorithm\":\"AES256\"}}]}"

echo Creating DynamoDB table for state locking: %TABLE%
aws dynamodb create-table --table-name %TABLE% --attribute-definitions AttributeName=LockID,AttributeType=S --key-schema AttributeName=LockID,KeyType=HASH --billing-mode PAY_PER_REQUEST --region %REGION% 2>nul
if %errorlevel% equ 0 (
    echo DynamoDB table created.
) else (
    echo DynamoDB table already exists, continuing.
)

echo.
echo Bootstrap complete.
echo   S3 bucket  : %BUCKET% (versioned, encrypted, private)
echo   DynamoDB   : %TABLE% (state locking)
echo.
echo Next step: cd terraform  then  terraform init  then  terraform apply

endlocal