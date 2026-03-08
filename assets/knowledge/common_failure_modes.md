# Common Failure Modes

This section describes **known situations where anomaly signals become ambiguous**
even when no clear defect is present.

These cases do not indicate model failure,
but rather conditions where **automated decisions are less reliable**.

## Typical situations

- Global or uneven lighting changes
- Camera focus drift or motion blur
- Background texture or material variations
- Product variants not represented in the normal training data

## Common characteristics

In these situations, the system may exhibit:

- Broad or diffuse anomaly responses rather than localized evidence
- Reduced decision reliability or stability
- Increased use of the **uncertain** label

These behaviors are **expected and intentional** under such conditions.

## Operational guidance

When these patterns appear frequently:

- Review imaging conditions (lighting, focus, alignment)
- Check for distribution shift in incoming data
- Confirm whether new product variants or backgrounds were introduced

These cases should trigger **data or environment review**,
not immediate model blame.

The system is designed to favor **conservative operation**:
when confidence is low, it escalates to human review rather than forcing automation.