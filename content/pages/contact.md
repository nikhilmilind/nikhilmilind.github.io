---
name: Contact
---

# Contact

Use the contact form below to send me an email. I will get back to you at the email you provide me.

<form id="contact-form" action="https://api.web3forms.com/submit" method="POST">
    <input type="hidden" name="access_key" value="bed5aa70-2860-43aa-9069-625691e2d65d">
    <div class="mb-4">
        <label for="contact-form-name" class="form-label">Name</label>
        <input type="text" class="form-control" name="name" id="contact-form-name" placeholder="Your name" required>
    </div>
    <div class="mb-4">
        <label for="contact-form-email" class="form-label">Email</label>
        <input type="email" class="form-control" name="email" id="contact-form-email" placeholder="Your email" required>
    </div>
    <div class="mb-4">
        <label for="contact-form-message" class="form-label">Message</label>
        <textarea name="message" class="form-control" id="contact-form-message" rows="5" required></textarea>
    </div>
    <div class="mb-4">
        <div class="h-captcha" data-captcha="true"></div>
    </div>
    <button type="submit" class="btn btn-primary">Submit</button>
</form>

<script>
const form = document.getElementById('contact-form');

form.addEventListener('submit', function(e) {

    const hCaptcha = form.querySelector('textarea[name=h-captcha-response]').value;

    if (!hCaptcha) {
        e.preventDefault();
        alert("Please fill out captcha field")
        return
    }
});
</script>
