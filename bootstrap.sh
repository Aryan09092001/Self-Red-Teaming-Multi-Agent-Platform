#!/bin/bash
set -e

REGION=${1:-us-east-1}
# S3 bucket names are globally unique across all AWS accounts, so the account id
# is appended to keep this one ours. Must match the backend block in terraform/main.tf.
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET="research-agent-tfstate-${ACCOUNT_ID}"
TABLE="research-agent-tf-locks"

echo "Creating S3 bucket: $BUCKET in region: $REGION"

if [ "$REGION" = "us-east-1" ]; then
  CREATE_ERR=$(aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" 2>&1 >/dev/null) || true
else
  CREATE_ERR=$(aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
    --create-bucket-configuration LocationConstraint="$REGION" 2>&1 >/dev/null) || true
fi

# Only "we already own it" is safe to ignore. Anything else (name taken by another
# account, bad credentials, wrong region) must stop the script, not be swallowed.
if [ -z "$CREATE_ERR" ]; then
  echo "Bucket created."
elif echo "$CREATE_ERR" | grep -q "BucketAlreadyOwnedByYou"; then
  echo "Bucket already exists and is owned by this account, continuing."
else
  echo "ERROR: could not create bucket $BUCKET" >&2
  echo "$CREATE_ERR" >&2
  exit 1
fi

echo "Enabling versioning on S3 bucket..."
aws s3api put-bucket-versioning \
  --bucket "$BUCKET" \
  --versioning-configuration Status=Enabled

echo "Blocking public access on S3 bucket..."
aws s3api put-public-access-block \
  --bucket "$BUCKET" \
  --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

echo "Enabling server-side encryption on S3 bucket..."
aws s3api put-bucket-encryption \
  --bucket "$BUCKET" \
  --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'

echo "Creating DynamoDB table for Terraform state locking: $TABLE"
aws dynamodb create-table \
  --table-name "$TABLE" \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region "$REGION" 2>/dev/null && echo "DynamoDB table created." || echo "DynamoDB table already exists, continuing."

echo ""
echo "Bootstrap complete."
echo "  S3 bucket  : $BUCKET (versioned, encrypted, private)"
echo "  DynamoDB   : $TABLE (state locking)"
echo ""
echo "Next step: cd terraform && terraform init && terraform apply"
