document.addEventListener("DOMContentLoaded", () => {
    console.log("Garbage Classifier loaded!");

    const fileInput = document.querySelector('input[type="file"]');
    const previewContainer = document.createElement("div");
    previewContainer.className = "preview-container";
    fileInput.parentNode.insertBefore(previewContainer, fileInput.nextSibling);

    const form = document.querySelector("form");
    const submitButton = form.querySelector("button[type='submit']");
    const loadingSpinner = document.createElement("div");
    loadingSpinner.className = "loading-spinner";
    loadingSpinner.style.display = "none";
    form.appendChild(loadingSpinner);

    // Preview the uploaded image
    fileInput.addEventListener("change", (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewContainer.innerHTML = `<img src="${e.target.result}" alt="Preview" class="preview-image">`;
            };
            reader.readAsDataURL(file);
        } else {
            previewContainer.innerHTML = "";
        }
    });

    // Show loading spinner on form submission
    form.addEventListener("submit", () => {
        loadingSpinner.style.display = "block";
        submitButton.disabled = true;
    });
});
