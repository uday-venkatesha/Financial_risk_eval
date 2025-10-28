# Financial Risk Assessment API Documentation

## Base URL
```
http://localhost:5000/api
```

## Authentication
Currently, the API does not require authentication. In production, implement OAuth 2.0 or API key authentication.

---

## Endpoints

### 1. Health Check

**GET** `/health`

Check API availability and status.

**Response:**
```json
{
  "status": "healthy",
  "service": "risk-assessment-api"
}
```

---

### 2. Predict Risk Score

**POST** `/api/predict`

Generate a risk prediction for a customer's loan application.

**Request Body:**
```json
{
  "customer_id": 123,
  "loan_amount": 50000.00,
  "loan_term_months": 36,
  "loan_purpose": "Home"
}
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| customer_id | integer | Yes | Unique customer identifier |
| loan_amount | float | Yes | Requested loan amount |
| loan_term_months | integer | No | Loan term (12, 24, 36, 48, 60, 72). Default: 36 |
| loan_purpose | string | No | Purpose of loan (Home, Auto, Personal, Education, Business) |

**Response (200 OK):**
```json
{
  "customer_id": 123,
  "loan_amount": 50000.00,
  "risk_score": 42.5,
  "risk_category": "Medium",
  "default_probability": 0.425,
  "recommendation": "Review",
  "timestamp": "2024-01-15T14:30:00"
}
```

**Risk Categories:**
- **Low** (0-30): Approve
- **Medium** (30-60): Review
- **High** (60-100): Reject

**Error Responses:**

400 Bad Request:
```json
{
  "error": "Missing required fields: customer_id"
}
```

404 Not Found:
```json
{
  "error": "Customer 123 not found"
}
```

503 Service Unavailable:
```json
{
  "error": "Model not available"
}
```

---

### 3. Get Existing Risk Score

**GET** `/api/risk-score/{customer_id}`

Retrieve the most recent risk score for a customer.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| customer_id | integer | Yes | Unique customer identifier (in path) |

**Response (200 OK):**
```json
{
  "customer_id": 123,
  "risk_score": 42.5,
  "risk_category": "Medium",
  "default_probability": 0.425,
  "prediction_date": "2024-01-15T14:30:00"
}
```

**Error Response (404):**
```json
{
  "error": "No risk score found for customer 123"
}
```

---

### 4. Get Model Metrics

**GET** `/api/model-metrics`

Retrieve current production model performance metrics.

**Response (200 OK):**
```json
{
  "model_name": "Risk Prediction Model",
  "model_version": "v1.0.0",
  "algorithm": "RandomForestClassifier",
  "accuracy": 0.873,
  "precision_score": 0.846,
  "recall": 0.792,
  "f1_score": 0.818,
  "auc_roc": 0.91,
  "training_date": "2024-01-10T10:00:00"
}
```

---

### 5. Batch Predictions

**POST** `/api/batch-predict`

Generate risk predictions for multiple customers at once.

**Request Body:**
```json
{
  "customers": [
    {
      "customer_id": 123,
      "loan_amount": 50000,
      "loan_term_months": 36
    },
    {
      "customer_id": 456,
      "loan_amount": 75000,
      "loan_term_months": 48
    }
  ]
}
```

**Limits:**
- Maximum 100 customers per request

**Response (200 OK):**
```json
{
  "total": 2,
  "successful": 2,
  "failed": 0,
  "results": [
    {
      "customer_id": 123,
      "risk_score": 42.5,
      "risk_category": "Medium",
      "status": "success"
    },
    {
      "customer_id": 456,
      "risk_score": 65.8,
      "risk_category": "High",
      "status": "success"
    }
  ]
}
```

**Error in Batch:**
```json
{
  "total": 2,
  "successful": 1,
  "failed": 1,
  "results": [
    {
      "customer_id": 123,
      "risk_score": 42.5,
      "risk_category": "Medium",
      "status": "success"
    },
    {
      "customer_id": 999,
      "status": "error",
      "message": "Customer not found"
    }
  ]
}
```

---

### 6. Get Feature Importance

**GET** `/api/feature-importance`

Retrieve feature importance scores from the model.

**Response (200 OK):**
```json
{
  "features": [
    {
      "feature": "credit_score",
      "importance": 0.245
    },
    {
      "feature": "debt_to_income_ratio",
      "importance": 0.187
    },
    {
      "feature": "loan_to_income_ratio",
      "importance": 0.142
    }
  ]
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input |
| 404 | Not Found - Resource doesn't exist |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Model not loaded |

## Rate Limiting
- 100 requests per minute per IP
- 1000 requests per hour per IP
- Batch predictions count as 1 request

## Example Usage

### cURL Examples

**Predict Risk:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": 123,
    "loan_amount": 50000,
    "loan_term_months": 36
  }'
```

**Get Risk Score:**
```bash
curl -X GET http://localhost:5000/api/risk-score/123
```

**Batch Predict:**
```bash
curl -X POST http://localhost:5000/api/batch-predict \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {"customer_id": 123, "loan_amount": 50000},
      {"customer_id": 456, "loan_amount": 75000}
    ]
  }'
```

### Python Example

```python
import requests

# Single prediction
response = requests.post(
    'http://localhost:5000/api/predict',
    json={
        'customer_id': 123,
        'loan_amount': 50000,
        'loan_term_months': 36
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"Risk Score: {result['risk_score']}")
    print(f"Category: {result['risk_category']}")
    print(f"Recommendation: {result['recommendation']}")
else:
    print(f"Error: {response.json()['error']}")
```

### JavaScript Example

```javascript
// Using fetch
fetch('http://localhost:5000/api/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    customer_id: 123,
    loan_amount: 50000,
    loan_term_months: 36
  })
})
.then(response => response.json())
.then(data => {
  console.log('Risk Score:', data.risk_score);
  console.log('Risk Category:', data.risk_category);
})
.catch(error => console.error('Error:', error));
```

## Response Time SLA
- Single predictions: < 200ms (95th percentile)
- Batch predictions: < 2s for 100 customers (95th percentile)

## Data Privacy
All customer data is encrypted in transit (HTTPS) and at rest. Predictions are logged for audit purposes and retained for 90 days.

## Versioning
API version is included in the model metadata. Breaking changes will be introduced in new API versions (e.g., `/api/v2/predict`).

## Support
For API issues or questions, contact: api-support@company.com