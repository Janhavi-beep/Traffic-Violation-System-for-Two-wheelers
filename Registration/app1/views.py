from django.shortcuts import render, redirect, HttpResponse
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.models import User
from django.conf import settings
from .yolo_predict import run_detection

import os

@csrf_exempt
def SignupPage(request):
    if request.method == 'POST':
        uname = request.POST.get('username')
        email = request.POST.get('email')
        pass1 = request.POST.get('password1')
        pass2 = request.POST.get('password2')

        if pass1 != pass2:
            return HttpResponse("Passwords do not match!")
        
        if User.objects.filter(username=uname).exists():
            return HttpResponse("Username already taken!")

        user = User.objects.create_user(uname, email, pass1)
        user.save()
        return redirect('login')

    return render(request, 'signup.html')

def LoginPage(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('pass')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('home')
        else:
            return HttpResponse("Invalid credentials")

    return render(request, 'login.html')

@login_required(login_url='login')
def HomePage(request):
    return render(request, 'home.html')

@login_required(login_url='login')
def dashboard(request):
    return render(request, 'dashboard.html')

@login_required(login_url='login')
def image_analysis_view(request):
    context = {}

    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']

        # Save the uploaded file to 'uploads' folder inside MEDIA_ROOT
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        fs = FileSystemStorage(location=upload_dir)
        filename = fs.save(image_file.name, image_file)

        # Full image path for detection and URL for display
        full_path = os.path.join(upload_dir, filename)
        file_url = f'uploads/{filename}'

        print("DEBUG — Final image path passed to model:", full_path)

        try:
            results = run_detection(full_path)

            # ✅ Updated class ID checks based on your dataset
            helmet_status = 'Yes' if any(d['class_id'] == 0 for d in results["detections"]) else 'No'
            number_plate_present = 'Yes' if any(d['class_id'] == 1 for d in results["detections"]) else 'No'
            passenger_count = sum(1 for d in results["detections"] if d['class_id'] == 4)  # Person count
            plate_number = results['ocr_results'][0]['text'] if results['ocr_results'] else 'Not detected'

            context = {
                "file_url": file_url,
                "helmet_status": helmet_status,
                "number_plate_present": number_plate_present,
                "passenger_count": passenger_count,
                "plate_number": plate_number,
                "detections": results["detections"],
                "ocr_results": results["ocr_results"]
            }
        except Exception as e:
            context = {"error": f"An error occurred during analysis: {str(e)}"}

    return render(request, 'image_analysis.html', context)

def image_upload_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        uploaded_file_url = fs.url(filename)
        return render(request, 'image_analysis.html', {'uploaded_file_url': uploaded_file_url})

def video_analysis(request):
    return render(request, 'video_analysis.html')

def Logoutview(request):
    logout(request)
    return redirect('login')
