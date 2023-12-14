from django.urls import path
from . import views
from .views import *



urlpatterns = [
    path('',views.home,name="home"),
    path('assignment1/',views.assignment1,name="assignment1"),
    path('assignment1_que2/',views.assignment1_que2,name="assignment1_que2"),
    path('assignment2/',views.assignment2,name="assignment2"),
    path('assignment3/',views.assignment3,name="assignment3"),
    path('assignment3/confuse_matrix/',views.assignment3_confuse_matrix,name="assignment3_confuse_matrix"),
    path('assignment4/',views.assignment4,name="assignment4"),
    path('assignment5/',views.assignment5,name="assignment5"),
    path('api/upload-csv/', CSVFileUploadView.as_view(), name='upload-csv-api'),
    path('assignment55/',RegressionClass.as_view(),name="regression"),
    path('hierarchical-clustering/<str:method>/', hierarchical_clustering, name='hierarchical_clustering'),
    path('kmeans_clustering/', kmeans_clustering, name='kmeans_clustering'),
    path('kmedoids_clustering/', kmedoids_clustering, name='kmedoids_clustering'),
    path('birch_clustering/', birch_clustering, name='birch_clustering'),
    path('dbscan_clustering/', dbscan_clustering, name='dbscan_clustering'),
    path('clustering_evaluation/',clustering_evaluation,name="clustering_evaluation"),
    path('generate_rules/',generate_rules,name="generate_rules"),
    path('generate_interesting_rules/',generate_interesting_rules,name="generate_interesting_rules"),
    path('crawl/', views.crawl, name='crawl'),
    path('calculate_pagerank/',views.calculate_pagerank,name="calculate_pagerank"),
    path('calculate_hits/',views.calculate_hits,name="calculate_hits"),
    path('crawler/', CrawlAPIView.as_view(), name='crawler'),
    path('generate-rules/', GenerateRulesView.as_view(), name='generate-rules'),



]