function CreatePDFfromHTML() {
    var HTML_Width = $("#content").width();
    var HTML_Height = $("#content").height();
    var top_left_margin = 15;
    var PDF_Width = HTML_Width + (top_left_margin * 2);
    var PDF_Height = (PDF_Width * 1.5) + (top_left_margin * 2);

    var canvas = document.createElement('canvas');
    var context = canvas.getContext('2d');
    canvas.width = HTML_Width;
    canvas.height = HTML_Height;

    // Set background color
    context.fillStyle = "#4c3228";
    context.fillRect(0, 0, canvas.width, canvas.height);

    // Apply styles to mimic the appearance of the HTML content
    context.font = "bold large Arial, sans-serif";
    context.fillStyle = "#FFD710";  // Text color

    // Render text content
    context.fillText("Pose Estimation Results", 15, 40);

    // Get the HTML content element
    var contentElement = document.getElementById('content');

    // Render HTML content to canvas
    html2canvas(contentElement, { canvas: canvas }).then(function (canvas) {
        var imgData = canvas.toDataURL("image/jpeg", 1.0);
        var pdf = new jsPDF('p', 'pt', [PDF_Width, PDF_Height]);

        // Add the image with the background color
        pdf.addImage(imgData, 'JPG', top_left_margin, top_left_margin, HTML_Width, HTML_Height);

        pdf.save("results.pdf");
    });
}
