import logo from './logo.svg';
import './App.css';




import React, { useState } from 'react';

const imageData = [
  { label: 'triangles', file: 'triangles.png' },
  { label: 'circles', file: 'circles.jpg' },
  { label: 'squares', file: 'squares.jpg' },
];

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageClick = async (image) => {
    setLoading(true);
    setResult(null);

    try {
      // Buscar a imagem local como blob
      const response = await fetch(`/${image.file}`);
      const blob = await response.blob();
      const file = new File([blob], image.file, { type: blob.type });

      // Montar o form-data
      const formData = new FormData();
      formData.append('image', file);
      formData.append('shape', image.label); // label enviada como referência

      // Enviar requisição à API Flask
      const res = await fetch('https://apicaptcha-xzmu.onrender.com/captcha/recognize', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (res.ok && data.shape) {
        const predicted = data.shape;
        const correct = predicted === image.label;

        setResult({
          expected: image.label,
          predicted: predicted,
          response: correct ? 'Correct!' : 'Incorrect!',
        });
      } else {
        setResult({ error: data.error || 'Unexpected response from API.' });
      }
    } catch (err) {
      setResult({ error: 'Failed to connect to the API.' });
    }

    setLoading(false);
  };

  return (
    <div style={{ padding: '2rem', textAlign: 'center' }}>
      <h1>Captcha Shape Recognition</h1>
      <p>Click on the shape to verify:</p>

      <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem' }}>
        {imageData.map((img) => (
          <div key={img.label}>
            <img
              src={`/${img.file}`}
              alt={img.label}
              width={100}
              style={{ cursor: 'pointer' }}
              onClick={() => handleImageClick(img)}
            />
            <p>{img.label}</p>
          </div>
        ))}
      </div>

      {loading && <p>Checking...</p>}

      {result && (
        <div style={{ marginTop: '1rem' }}>
          {result.error ? (
            <p style={{ color: 'red' }}>Error: {result.error}</p>
          ) : (
            <>
              <p>Expected: <strong>{result.expected}</strong></p>
              <p>Predicted: <strong>{result.predicted}</strong></p>
              <p>Result: <strong>{result.response}</strong></p>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
