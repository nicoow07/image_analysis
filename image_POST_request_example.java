URL url = new URL(imgServerIP);
                    HttpURLConnection imageServerConnection = (HttpURLConnection) url.openConnection();
                    imageServerConnection.setRequestMethod("POST");
                    imageServerConnection.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");
                    //imageServerConnection.setRequestProperty("Accept", "application/json");
                    imageServerConnection.setRequestProperty("User-Agent", "Mozilla/5.0");
                    imageServerConnection.setDoOutput(true);

                    //Send string test - WORKING
                    /*String jsonInputString = "{name: Upendra, job: Programmer}";
                    OutputStream os = imageServerConnection.getOutputStream();
                    byte[] input = jsonInputString.getBytes("utf-8");
                    os.write(input, 0, input.length);
                    os.flush();
                    os.close();*/

                    //Send image
                    String imgFileName = String.valueOf(Calendar.getInstance().getTimeInMillis()) + "_shot.jpg";
                    String boundary =  "*****";
                    imageServerConnection.setRequestProperty("Content-Type", "multipart/form-data;boundary="+boundary);
                    DataOutputStream os = new DataOutputStream(imageServerConnection.getOutputStream());
                    os.writeBytes("--" + boundary + "\r\n");
                    os.writeBytes("Content-Disposition: form-data; name=\"image\";filename=\"" + imgFileName + "\"\r\n"); // uploaded_file_name is the Name of the File to be uploaded
                    os.writeBytes("\r\n");



                    InputStream imgInputStream = new ByteArrayInputStream(rotatedCaptureBytes);
                    int maxBufferSize = 1024*1024;
                    int bytesAvailable = imgInputStream.available();
                    int bufferSize = Math.min(bytesAvailable, maxBufferSize);
                    int bytesRead = 1;
                    byte[] buffer = new byte[bufferSize];
                    bytesRead = imgInputStream.read(buffer, 0, bufferSize);
                    while (bytesRead > 0){
                        os.write(rotatedCaptureBytes, 0, bufferSize);
                        bytesAvailable = imgInputStream.available();
                        bufferSize = Math.min(bytesAvailable, maxBufferSize);
                        bytesRead = imgInputStream.read(buffer, 0, bufferSize);
                    }
                    os.writeBytes("\r\n");
                    os.writeBytes(("--" + boundary + "--" + "\r\n"));
                    imgInputStream.close();
                    os.flush();
                    os.close();

                    //Response
                    int responseCode = imageServerConnection.getResponseCode();//THIS LINE SENDS THE POST REQUEST...
                    if (responseCode == HttpURLConnection.HTTP_OK) { //success
                        BufferedReader in = new BufferedReader(new InputStreamReader(imageServerConnection.getInputStream()));
                        String inputLine;
                        StringBuffer response = new StringBuffer();

                        while ((inputLine = in.readLine()) != null) {
                            response.append(inputLine);
                        }
                        in.close();

                        // print result
                        System.out.println(response.toString());
                    } else {
                        System.out.println("POST request did not work. Request code: " + String.valueOf(responseCode));
                    } 
