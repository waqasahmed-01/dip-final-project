// ---------- IMAGE UPLOAD ----------
const imageUpload = document.getElementById('imageUpload');
const uploadedImage = document.getElementById('uploadedImage');
const resultImage = document.getElementById('resultImage');

imageUpload.addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  // Show preview instantly
  uploadedImage.src = URL.createObjectURL(file);

  const formData = new FormData();
  formData.append('image', file);

  try {
    const response = await fetch('http://127.0.0.1:5000/face-detect', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) throw new Error('Face detection failed.');

    const blob = await response.blob();
    const imageUrl = URL.createObjectURL(blob);
    resultImage.src = imageUrl;
  } catch (error) {
    alert('Error: ' + error.message);
  }
});
