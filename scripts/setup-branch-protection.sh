#!/bin/bash
# Branch Protection Setup for Business Agent Management System
# 
# This script configures GitHub branch protection rules to work with our
# incremental CI workflow and quality gate system.
#
# Prerequisites:
# - GitHub CLI (gh) installed and authenticated
# - Repository write access
# - Enable branch protections in repository settings
#
# Usage:
#   ./scripts/setup-branch-protection.sh [--dry-run] [--branch BRANCH]

set -euo pipefail

# Configuration
DEFAULT_BRANCH="main"
BRANCH="${1:-$DEFAULT_BRANCH}"
DRY_RUN=false
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--dry-run] [--branch BRANCH]"
      echo ""
      echo "Options:"
      echo "  --dry-run     Show what would be done without making changes"
      echo "  --branch      Branch to protect (default: main)"
      echo "  --help        Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo -e "${BLUE}🔒 Setting up branch protection for: ${YELLOW}$BRANCH${NC}"
echo "Repository: $(gh repo view --json nameWithOwner -q .nameWithOwner)"
echo ""

# Check if gh CLI is authenticated
if ! gh auth status &>/dev/null; then
    echo -e "${RED}❌ GitHub CLI is not authenticated${NC}"
    echo "Please run: gh auth login"
    exit 1
fi

# Check if we have the necessary permissions
if ! gh api repos/:owner/:repo --silent; then
    echo -e "${RED}❌ Cannot access repository or insufficient permissions${NC}"
    exit 1
fi

# Branch protection configuration
PROTECTION_CONFIG=$(cat <<EOF
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "final-validation-gate",
      "incremental-validation (3.8)",
      "incremental-validation (3.11)",
      "quality-gate-enforcement (3.11)",
      "comprehensive-testing (3.8)",
      "comprehensive-testing (3.11)",
      "security-scan"
    ]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true,
    "require_last_push_approval": false
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false,
  "required_conversation_resolution": true,
  "lock_branch": false,
  "allow_fork_syncing": true
}
EOF
)

echo -e "${BLUE}📋 Branch Protection Configuration:${NC}"
echo "$PROTECTION_CONFIG" | jq .

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo -e "${YELLOW}🔍 DRY RUN MODE - No changes will be made${NC}"
    echo ""
    echo "The following command would be executed:"
    echo "gh api repos/:owner/:repo/branches/$BRANCH/protection --method PUT --input <protection_config>"
    echo ""
    echo -e "${GREEN}✅ Dry run completed successfully${NC}"
    exit 0
fi

echo ""
echo -e "${YELLOW}⚠️  This will update branch protection rules for '$BRANCH'${NC}"
echo "Continue? (y/N)"
read -r response

if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}🛑 Aborted by user${NC}"
    exit 0
fi

echo ""
echo -e "${BLUE}🔧 Applying branch protection rules...${NC}"

# Apply branch protection
if echo "$PROTECTION_CONFIG" | gh api "repos/:owner/:repo/branches/$BRANCH/protection" --method PUT --input -; then
    echo -e "${GREEN}✅ Branch protection rules applied successfully${NC}"
else
    echo -e "${RED}❌ Failed to apply branch protection rules${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}📊 Current branch protection status:${NC}"
gh api "repos/:owner/:repo/branches/$BRANCH/protection" | jq '{
  required_status_checks: .required_status_checks.contexts,
  required_reviews: .required_pull_request_reviews.required_approving_review_count,
  enforce_admins: .enforce_admins.enabled,
  dismiss_stale_reviews: .required_pull_request_reviews.dismiss_stale_reviews,
  require_code_owner_reviews: .required_pull_request_reviews.require_code_owner_reviews
}'

echo ""
echo -e "${GREEN}🎉 Branch protection setup completed!${NC}"
echo ""
echo -e "${BLUE}📖 Next steps:${NC}"
echo "1. Verify CI workflows are working correctly"
echo "2. Test pull request workflow with the new protections"
echo "3. Update team documentation about the new requirements"
echo "4. Consider setting up CODEOWNERS file for code review requirements"

echo ""
echo -e "${BLUE}💡 Useful commands:${NC}"
echo "- View protection status: gh api repos/:owner/:repo/branches/$BRANCH/protection"
echo "- List required status checks: gh api repos/:owner/:repo/branches/$BRANCH/protection | jq .required_status_checks.contexts"
echo "- Remove protection: gh api repos/:owner/:repo/branches/$BRANCH/protection --method DELETE"