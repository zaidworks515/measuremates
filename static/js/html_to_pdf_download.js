function CreatePDFfromHTML() {
    var contentElement = document.getElementById('content');
    
    // Calculate dimensions based on rendered content
    var HTML_Width = contentElement.offsetWidth;
    var HTML_Height = contentElement.offsetHeight;
    
    // Create canvas with dynamic dimensions
    var canvas = document.createElement('canvas');
    var context = canvas.getContext('2d');
    canvas.width = HTML_Width ;
    canvas.height = HTML_Height - 50;

    // Set background color
    context.fillStyle = "#FFFFFF"; // White background
    context.fillRect(0, 0, canvas.width, canvas.height);

    // Apply styles to mimic the appearance of the HTML content
    context.font = "bold 20px Arial, sans-serif";
    context.fillStyle = "#333"; // Text color

    // Render text content
    context.fillText("Results", 15, 40);

    // Render HTML content to canvas
    html2canvas(contentElement, { canvas: canvas }).then(function (canvas) {
        var imgData = canvas.toDataURL("image/jpeg", 1.0);
        var pdf = new jsPDF('p', 'pt', [HTML_Width, HTML_Height]);

        // Add the image with the background color
        pdf.addImage(imgData, 'JPG', 0, 0, HTML_Width, HTML_Height);

        // Save the PDF with the file name 'results.pdf'
        pdf.save("results.pdf");
    });
}
