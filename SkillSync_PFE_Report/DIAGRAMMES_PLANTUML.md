# Codes PlantUML pour les Diagrammes SkillSync

Ce fichier contient tous les codes PlantUML nécessaires pour générer les diagrammes du rapport.
Utilisez https://www.plantuml.com/plantuml/ pour générer les images PNG.

---

## 1. Diagramme de Cas d'Utilisation Global

```plantuml
@startuml usecase_global
left to right direction
skinparam packageStyle rectangle
skinparam actorStyle awesome

actor "Visiteur" as V
actor "Candidat" as C
actor "Administrateur" as A

rectangle "SkillSync" {
    usecase "S'inscrire" as UC1
    usecase "Se connecter" as UC2
    usecase "Se déconnecter" as UC3
    usecase "Télécharger CV" as UC4
    usecase "Analyser CV" as UC5
    usecase "Voir analyse détaillée" as UC6
    usecase "Générer portfolio" as UC7
    usecase "Personnaliser portfolio" as UC8
    usecase "Télécharger portfolio" as UC9
    usecase "Rechercher emplois" as UC10
    usecase "Voir matching score" as UC11
    usecase "Sauvegarder offre" as UC12
    usecase "Consulter guidance carrière" as UC13
    usecase "Voir recommandations" as UC14
    usecase "Gérer profil" as UC15
    usecase "Gérer utilisateurs" as UC16
    usecase "Voir statistiques" as UC17
}

V --> UC1
V --> UC2

C --> UC2
C --> UC3
C --> UC4
C --> UC5
C --> UC6
C --> UC7
C --> UC8
C --> UC9
C --> UC10
C --> UC11
C --> UC12
C --> UC13
C --> UC14
C --> UC15

A --> UC16
A --> UC17

UC5 .> UC4 : <<include>>
UC6 .> UC5 : <<include>>
UC8 .> UC7 : <<extend>>
UC11 .> UC10 : <<include>>
@enduml
```

Fichier de sortie: `img/use_case_global.png`

---

## 2. Diagramme de Classes Global

```plantuml
@startuml class_diagram_global
skinparam classAttributeIconSize 0

class User {
    -id: int
    -email: string
    -username: string
    -hashed_password: string
    -full_name: string
    -is_active: boolean
    -created_at: datetime
    +register()
    +login()
    +logout()
}

class RefreshToken {
    -id: int
    -token: string
    -user_id: int
    -expires_at: datetime
    -revoked: boolean
    +create()
    +revoke()
    +is_valid()
}

class CVAnalysis {
    -id: int
    -user_id: int
    -filename: string
    -raw_text: string
    -analysis_result: json
    -ats_score: float
    -created_at: datetime
    +analyze()
    +extract_skills()
    +calculate_ats_score()
}

class Skill {
    -id: int
    -name: string
    -category: string
    -confidence: float
    -source: string
}

class Portfolio {
    -id: int
    -user_id: int
    -template: string
    -color_scheme: string
    -content: json
    -file_path: string
    -created_at: datetime
    +generate()
    +customize()
    +export()
}

class JobSearch {
    -id: int
    -user_id: int
    -query: string
    -location: string
    -filters: json
    -results: json
    -created_at: datetime
    +search()
    +filter()
    +match_with_cv()
}

class JobOffer {
    -id: int
    -title: string
    -company: string
    -location: string
    -description: string
    -salary: string
    -source: string
    -matching_score: float
}

class CareerGuidance {
    -id: int
    -user_id: int
    -current_skills: json
    -target_role: string
    -recommendations: json
    -learning_path: json
    -created_at: datetime
    +analyze_career()
    +generate_recommendations()
    +suggest_learning_path()
}

class Recommendation {
    -id: int
    -type: string
    -title: string
    -description: string
    -priority: string
    -estimated_time: string
    -resources: json
}

User "1" -- "*" RefreshToken
User "1" -- "*" CVAnalysis
User "1" -- "*" Portfolio
User "1" -- "*" JobSearch
User "1" -- "*" CareerGuidance
CVAnalysis "1" -- "*" Skill
JobSearch "1" -- "*" JobOffer
CareerGuidance "1" -- "*" Recommendation
@enduml
```

Fichier de sortie: `img/class_diagram_global.png`

---

## 3. Diagramme de Séquence - Connexion

```plantuml
@startuml sequence_login
actor Utilisateur
participant "Frontend\nReact" as FE
participant "Backend\nFastAPI" as BE
database "PostgreSQL" as DB

Utilisateur -> FE : Saisir email/mot de passe
FE -> BE : POST /api/v1/auth/login
BE -> DB : Rechercher utilisateur par email
DB --> BE : Données utilisateur
BE -> BE : Vérifier mot de passe (bcrypt)
alt Authentification réussie
    BE -> BE : Générer access_token (15min)
    BE -> BE : Générer refresh_token (7j)
    BE -> DB : Sauvegarder refresh_token
    BE --> FE : {access_token, refresh_token, user}
    FE -> FE : Stocker tokens (localStorage)
    FE --> Utilisateur : Redirection Dashboard
else Authentification échouée
    BE --> FE : 401 Unauthorized
    FE --> Utilisateur : Message d'erreur
end
@enduml
```

Fichier de sortie: `img/sequence_login.png`

---

## 4. Diagramme de Séquence - Analyse CV

```plantuml
@startuml sequence_cv_analysis
actor Utilisateur
participant "Frontend" as FE
participant "CV Router" as API
participant "CV Analyzer" as CA
participant "BERT NER" as NER
participant "ATS Scorer" as ATS
database "PostgreSQL" as DB

Utilisateur -> FE : Upload CV (PDF/DOCX)
FE -> API : POST /api/v1/cv/analyze
API -> CA : parse_document(file)
CA -> CA : extract_text()
CA -> NER : extract_entities(text)
NER -> NER : tokenize()
NER -> NER : predict_entities()
NER --> CA : entities[skills, tools, tech]
CA -> CA : categorize_skills()
CA -> ATS : calculate_score(cv_data)
ATS -> ATS : keywords_score()
ATS -> ATS : structure_score()
ATS -> ATS : format_score()
ATS --> CA : ats_score + details
CA -> CA : generate_recommendations()
CA --> API : analysis_result
API -> DB : save_analysis()
API --> FE : CVAnalysisResponse
FE --> Utilisateur : Afficher résultats
@enduml
```

Fichier de sortie: `img/sequence_cv_analysis.png`

---

## 5. Diagramme de Séquence - Génération Portfolio

```plantuml
@startuml sequence_portfolio
actor Utilisateur
participant "Frontend" as FE
participant "Portfolio API" as API
participant "Template Engine" as TE
participant "Asset Manager" as AM
participant "ZIP Generator" as ZIP

Utilisateur -> FE : Sélectionner template + couleurs
FE -> API : POST /api/v1/portfolio/generate
API -> API : Récupérer données CV analysé
API -> TE : render_template(template, data, colors)
TE -> TE : Générer HTML
TE -> TE : Générer CSS personnalisé
TE -> TE : Générer JavaScript
TE --> API : rendered_files
API -> AM : collect_assets(fonts, icons)
AM --> API : assets[]
API -> ZIP : create_package(files, assets)
ZIP -> ZIP : Compression
ZIP --> API : portfolio.zip
API --> FE : download_url
FE --> Utilisateur : Télécharger ZIP
@enduml
```

Fichier de sortie: `img/sequence_portfolio.png`

---

## 6. Diagramme de Séquence - Matching CV-Offre

```plantuml
@startuml sequence_matching
actor Utilisateur
participant "Frontend" as FE
participant "Job Search API" as API
participant "External APIs" as EXT
participant "Semantic Matcher" as SM
participant "Sentence-Transformers" as ST

Utilisateur -> FE : Rechercher "Python Developer Paris"
FE -> API : GET /api/v1/jobs/search?q=...
API -> EXT : Adzuna API
API -> EXT : The Muse API
API -> EXT : RemoteOK API
EXT --> API : jobs[]
API -> API : Fusionner et dédupliquer
Utilisateur -> FE : Voir matching pour offre X
FE -> API : POST /api/v1/jobs/match
API -> SM : calculate_match(cv, job)
SM -> ST : encode(cv_text)
ST --> SM : cv_embedding[768]
SM -> ST : encode(job_description)
ST --> SM : job_embedding[768]
SM -> SM : cosine_similarity()
SM -> SM : analyze_skills()
SM -> SM : generate_explanation()
SM --> API : MatchResult
API --> FE : {score, explanation, details}
FE --> Utilisateur : Afficher résultat
@enduml
```

Fichier de sortie: `img/sequence_matching.png`

---

## 7. Diagramme de Séquence - Guidance Carrière

```plantuml
@startuml sequence_career
actor Utilisateur
participant "Frontend" as FE
participant "Career API" as API
participant "Guidance Engine" as GE
participant "Skill Taxonomy" as TAX
participant "Learning DB" as LDB

Utilisateur -> FE : Définir objectif "Senior Developer"
FE -> API : POST /api/v1/career/analyze
API -> API : Récupérer skills du CV analysé
API -> GE : analyze_career(skills, target)
GE -> TAX : get_required_skills("Senior Developer")
TAX --> GE : required_skills[]
GE -> GE : identify_gaps()
GE -> GE : prioritize_gaps()
GE -> LDB : find_courses(gaps)
LDB --> GE : courses[]
GE -> LDB : find_certifications(gaps)
LDB --> GE : certifications[]
GE -> GE : create_learning_path()
GE -> GE : estimate_time()
GE --> API : CareerAnalysis
API --> FE : {gaps, recommendations, path, time}
FE --> Utilisateur : Afficher guidance
@enduml
```

Fichier de sortie: `img/sequence_career.png`

---

## 8. Diagramme d'Architecture Technique

```plantuml
@startuml architecture
!define RECTANGLE class

skinparam componentStyle rectangle

package "Frontend (React + TypeScript)" {
    [Pages] --> [Components]
    [Components] --> [Services]
    [Services] --> [API Client (Axios)]
}

package "Backend (FastAPI)" {
    [Routers] --> [Services]
    [Services] --> [Repositories]
    [Repositories] --> [Models]
    [Services] --> [ML Modules]
}

package "ML/AI Layer" {
    [BERT NER Model]
    [Sentence Transformers]
    [spaCy NLP]
}

package "Data Layer" {
    database "PostgreSQL" as DB
    database "File Storage" as FS
}

[API Client (Axios)] --> [Routers] : HTTP/REST
[ML Modules] --> [BERT NER Model]
[ML Modules] --> [Sentence Transformers]
[ML Modules] --> [spaCy NLP]
[Repositories] --> DB
[Services] --> FS
@enduml
```

Fichier de sortie: `img/architecture.png`

---

## 9. Diagramme de Cas d'Utilisation - Authentification

```plantuml
@startuml usecase_auth
left to right direction
actor "Visiteur" as V
actor "Utilisateur" as U

rectangle "Module Authentification" {
    usecase "S'inscrire" as UC1
    usecase "Se connecter" as UC2
    usecase "Se déconnecter" as UC3
    usecase "Rafraîchir token" as UC4
    usecase "Réinitialiser mot de passe" as UC5
}

V --> UC1
V --> UC2
U --> UC2
U --> UC3
U --> UC4
U --> UC5

UC2 .> UC4 : <<extend>>
@enduml
```

Fichier de sortie: `img/usecase_auth.png`

---

## 10. Diagramme de Cas d'Utilisation - Analyse CV

```plantuml
@startuml usecase_cv
left to right direction
actor "Candidat" as U

rectangle "Module Analyse CV" {
    usecase "Télécharger CV" as UC1
    usecase "Analyser CV" as UC2
    usecase "Voir compétences extraites" as UC3
    usecase "Voir score ATS" as UC4
    usecase "Voir recommandations" as UC5
    usecase "Sauvegarder analyse" as UC6
    usecase "Historique analyses" as UC7
}

U --> UC1
U --> UC2
U --> UC3
U --> UC4
U --> UC5
U --> UC6
U --> UC7

UC2 .> UC1 : <<include>>
UC3 .> UC2 : <<include>>
UC4 .> UC2 : <<include>>
UC5 .> UC2 : <<include>>
@enduml
```

Fichier de sortie: `img/usecase_cv.png`

---

## 11. Diagramme de Classes - Module Auth

```plantuml
@startuml class_auth
class User {
    -id: int
    -email: str
    -username: str
    -hashed_password: str
    -full_name: str
    -is_active: bool
    -created_at: datetime
    +verify_password(password): bool
}

class RefreshToken {
    -id: int
    -token: str
    -user_id: int
    -expires_at: datetime
    -revoked: bool
    +is_valid(): bool
    +revoke(): void
}

class AuthService {
    +register(user_data): User
    +login(credentials): TokenPair
    +logout(token): void
    +refresh(refresh_token): TokenPair
    +get_current_user(token): User
}

class JWTHandler {
    -secret_key: str
    -algorithm: str
    +create_access_token(data): str
    +create_refresh_token(data): str
    +verify_token(token): dict
}

User "1" -- "*" RefreshToken
AuthService --> User
AuthService --> RefreshToken
AuthService --> JWTHandler
@enduml
```

Fichier de sortie: `img/class_auth.png`

---

## 12. Diagramme de Gantt - Planification

```plantuml
@startgantt
project starts 2024-09-01

[Sprint 1: Auth] starts 2024-09-01 and lasts 14 days
[Sprint 2: CV Analysis] starts after [Sprint 1: Auth] and lasts 14 days
[Sprint 3: Portfolio] starts after [Sprint 2: CV Analysis] and lasts 14 days
[Sprint 4: Job Matching] starts after [Sprint 3: Portfolio] and lasts 14 days
[Sprint 5: Career Guidance] starts after [Sprint 4: Job Matching] and lasts 14 days
[Sprint 6: Dashboard] starts after [Sprint 5: Career Guidance] and lasts 14 days

[Release 1] happens at [Sprint 2: CV Analysis]'s end
[Release 2] happens at [Sprint 4: Job Matching]'s end
[Release 3] happens at [Sprint 6: Dashboard]'s end
@endgantt
```

Fichier de sortie: `img/gantt.png`

---

## Instructions de génération

1. Allez sur https://www.plantuml.com/plantuml/
2. Collez chaque code PlantUML
3. Téléchargez l'image PNG
4. Renommez selon le nom de fichier indiqué
5. Placez dans le dossier `img/` du rapport

