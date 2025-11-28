const Landing = () => (
  <header className="min-h-[60vh] flex flex-col justify-center items-center px-6 text-center">
    <h1 className="text-4xl md:text-6xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 to-blue-500">
      Helmet & Seat-belt Detection
    </h1>
    <p className="mt-4 max-w-xl text-lg text-gray-300">
      Upload an image or capture from webcam to verify rider safety
      in real-time using our YOLOv10 model.
    </p>
  </header>
);
export default Landing;
