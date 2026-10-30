import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()
import os

configuration = sib_api_v3_sdk.Configuration()
configuration.api_key['api-key'] = os.getenv("BREVO_API_KEY")

VERIFIED_SENDER = os.getenv("SENDER", "harsh273003@gmail.com")


@tool
def send_email(to: str, subject: str, body: str):
    """Send an email. Parameters 'recipient', 'subject', and 'body' must be valid medium strings without raw linebreaks. and no long body"""

    api_instance = sib_api_v3_sdk.TransactionalEmailsApi(
        sib_api_v3_sdk.ApiClient(configuration)
    )

    send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
        to=[{"email": to}],
        sender={"email": VERIFIED_SENDER, "name": "Veronica"},
        subject=subject,
        text_content=body,
    )

    try:
        api_instance.send_transac_email(send_smtp_email)
        return "Email sent successfully"
    except ApiException as e:
        return f"Failed to send email: {e}"


if __name__ == "__main__":
    result = send_email.invoke({
        "to": "thakur273003@gmail.com",
        "subject": "Veronica Test",
        "body": "This is a test email from Veronica via Brevo."
    })

    print(result)